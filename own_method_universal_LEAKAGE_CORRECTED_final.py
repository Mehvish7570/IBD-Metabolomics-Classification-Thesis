
# ============================================================================

# UNIVERSAL PIPELINE - Own final combined methodology, dataset-agnostic

# Auto-detects sample ID, disease label, and (optionally) compound-mapping

# columns, then runs the complete adaptive final approach pipeline

# (leakage-safe preprocessing, small/large sample branch, grid search

# RF+XGBoost, Boruta, 4-classifier comparison).

# ============================================================================



import argparse

import re

import sys



import numpy as np

import pandas as pd

import xgboost as xgb

from boruta import BorutaPy

from imblearn.over_sampling import SMOTE

from scipy.special import softmax

from scipy.stats import kruskal, mannwhitneyu

from sklearn.ensemble import IsolationForest, RandomForestClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.multitest import multipletests





SAMPLE_ID_PATTERNS = [r'^sample', r'patient.?id', r'^id$', r'subject.?id']

LABEL_PATTERNS = [r'study.?group', r'diagnosis', r'disease', r'^class$', r'^label$', r'condition', r'health.?status']

MAPPING_ID_PATTERNS = [r'cluster.?id', r'compound.?id', r'feature.?id', r'metabolite.?id', r'^id$']

IDENTITY_PATTERNS = [
    r'^hmdb(?:.?id|.?accession)?$', r'^kegg(?:.?id|.?compound)?$',
    r'^compound.?name$', r'^metabolite.?name$',
    r'^putative.?chemical.?class$', r'^chemical.?class$',
]





def parse_args():

    parser = argparse.ArgumentParser(description="Universal untargeted-omics classification pipeline")

    parser.add_argument("--features_file", required=True, help="Path to the feature/metabolite TSV file")

    parser.add_argument("--metadata_file", required=True, help="Path to the sample metadata TSV file")

    parser.add_argument("--mapping_file", default=None, help="Optional path to a compound-mapping file (HMDB/KEGG identity)")

    parser.add_argument("--id_mass_file", default=None, help="Optional path to a simple 2-column CSV (feature ID, mass) - used to auto-generate HMDB/KEGG identities from scratch when the feature names themselves do not contain an embedded mass value")

    parser.add_argument(
        "--ion_mode",
        choices=("auto", "positive", "negative"),
        default="auto",
        help=("Ionization mode for mass matching. 'auto' infers positive/negative "
              "from feature IDs containing '-pos' or '-neg'."),
    )

    parser.add_argument("--mass_tolerance_ppm", type=float, default=10.0,
                        help="Mass tolerance used to report HMDB candidates (default: 10 ppm)")

    parser.add_argument("--hmdb_cache", default=HMDB_CACHE_PATH,
                        help="CSV export of HMDB containing accession and monoisotopic molecular weight")

    parser.add_argument("--results_dir", default="/rds/projects/g/guellili-slr-automation/mehvish/thesis/results", help="Directory to save results CSV")

    parser.add_argument("--dataset_name", default="Universal_Test", help="Label for this run, used in printed output")

    return parser.parse_args()





def detect_column(columns, patterns, purpose):

    for col in columns:

        col_clean = col.strip().lower().replace(" ", "_")

        for pattern in patterns:

            if re.search(pattern, col_clean):

                print(f"Auto-detected {purpose} column: '{col}'", flush=True)

                return col

    print(f"ERROR: could not auto-detect {purpose} column among: {list(columns)}", flush=True)

    sys.exit(1)





def load_data(features_file, metadata_file, dataset_name):

    mtb = pd.read_csv(features_file, sep="\t")

    if mtb.shape[1] <= 1:

        mtb = pd.read_csv(features_file, sep=None, engine="python")

    metadata = pd.read_csv(metadata_file, sep="\t")

    if metadata.shape[1] <= 1:

        metadata = pd.read_csv(metadata_file, sep=None, engine="python")

    print(f"Loaded {dataset_name}: {mtb.shape[0]} samples", flush=True)

    print(f"Raw column headers (first 5): {list(mtb.columns)[:5]}", flush=True)

    print(f"Raw metadata headers: {list(metadata.columns)}", flush=True)

    return mtb, metadata





def detect_sample_and_label_columns(mtb, metadata):

    sample_col = detect_column(mtb.columns, SAMPLE_ID_PATTERNS, "sample ID (feature file)")

    label_col = detect_column(metadata.columns, LABEL_PATTERNS, "disease label (metadata file)")

    meta_sample_col = detect_column(metadata.columns, SAMPLE_ID_PATTERNS, "sample ID (metadata file)")

    return sample_col, label_col, meta_sample_col





HMDB_CACHE_PATH = "hmdb_metabolites_cache.csv"



# ============================================================================

# COMPOUND MAPPING GENERATION — builds a full HMDB/KEGG mapping file from

# scratch, using ONLY each feature's m/z value and its compound/feature ID.

# No pre-existing mapping file, HMDB ID, or KEGG ID is used as input —

# everything is looked up fresh against the HMDB reference database

# (hmdb_metabolites_cache.csv). This works for ANY dataset with a parseable

# mass value embedded in (or alongside) its feature names, not just one

# specific dataset.

# ============================================================================



# Common singly charged LC-MS adducts. To recover a neutral mass from a

# compound's neutral monoisotopic mass from a measured m/z, we subtract

# the adduct's correction: neutral_mass = measured_mz - correction.

ADDUCTS_BY_POLARITY = {
    "positive": {
        "[M+H]+": 1.007276,
        "[M+Na]+": 22.989218,
        "[M+K]+": 38.963158,
        "[M+NH4]+": 18.034164,
        "[M-H2O+H]+": -17.002739,
    },
    "negative": {
        "[M-H]-": -1.007276,
        "[M+Cl]-": 34.969402,
        "[M+FA-H]-": 44.997655,
        "[M+CH3COO]-": 59.013305,
    },
}

# Rank the canonical protonated/deprotonated ions before alternative adducts.
PRIMARY_ADDUCTS = {"[M+H]+", "[M-H]-"}





def _load_hmdb_reference(hmdb_cache_path=HMDB_CACHE_PATH):
    """Load and mass-sort the local HMDB reference used for reproducible matching."""
    hmdb_df = pd.read_csv(hmdb_cache_path, low_memory=False)
    required = {"accession", "monoisotopic_molecular_weight"}
    missing = required.difference(hmdb_df.columns)
    if missing:
        raise ValueError(f"HMDB cache is missing required columns: {sorted(missing)}")

    hmdb_df["monoisotopic_molecular_weight"] = pd.to_numeric(
        hmdb_df["monoisotopic_molecular_weight"], errors="coerce"
    )
    hmdb_df = (
        hmdb_df.dropna(subset=["accession", "monoisotopic_molecular_weight"])
        .sort_values("monoisotopic_molecular_weight", kind="mergesort")
        .reset_index(drop=True)
    )
    if hmdb_df.empty:
        raise ValueError("HMDB cache contains no usable monoisotopic masses")
    return hmdb_df


def _infer_polarity(feature_id, ion_mode="auto"):
    """Use an explicit mode, or infer '-pos'/'-neg' from a feature ID."""
    if ion_mode in ("positive", "negative"):
        return ion_mode
    text = str(feature_id).lower()
    if re.search(r"(?:^|[-_])pos(?:itive)?(?:[-_:]|$)", text):
        return "positive"
    if re.search(r"(?:^|[-_])neg(?:ative)?(?:[-_:]|$)", text):
        return "negative"
    return "unknown"


def _hmdb_candidates(mz, hmdb_df, tolerance_ppm=10.0, polarity="unknown"):
    """Return every HMDB/adduct candidate inside the ppm window.

    These are exact-mass candidates, not confirmed metabolite identifications.
    """
    mz = float(mz)
    if not np.isfinite(mz) or mz <= 0:
        return []

    if polarity in ADDUCTS_BY_POLARITY:
        adducts = ADDUCTS_BY_POLARITY[polarity]
    else:
        adducts = {
            **ADDUCTS_BY_POLARITY["positive"],
            **ADDUCTS_BY_POLARITY["negative"],
        }

    masses = hmdb_df["monoisotopic_molecular_weight"].to_numpy()
    tolerance_da = mz * float(tolerance_ppm) / 1e6
    candidates = []

    for adduct, correction in adducts.items():
        target_neutral_mass = mz - correction
        if target_neutral_mass <= 0:
            continue
        left = np.searchsorted(masses, target_neutral_mass - tolerance_da, side="left")
        right = np.searchsorted(masses, target_neutral_mass + tolerance_da, side="right")

        for idx in range(left, right):
            row = hmdb_df.iloc[idx]
            theoretical_mz = float(row["monoisotopic_molecular_weight"]) + correction
            ppm_error = (mz - theoretical_mz) / theoretical_mz * 1e6
            if abs(ppm_error) <= tolerance_ppm:
                def value(column):
                    return row[column] if column in row.index and pd.notna(row[column]) else "NA"

                candidates.append({
                    "HMDB": row["accession"],
                    # KEGG is an HMDB cross-reference; it is not independently mass-matched.
                    "KEGG": value("kegg_id"),
                    "Compound.Name": value("name"),
                    "Chemical.Formula": value("chemical_formula"),
                    "CAS.Registry.Number": value("cas_registry_number"),
                    "ChEBI.ID": value("chebi_id"),
                    "PubChem.CID": value("pubchem_compound_id"),
                    "Candidate.Adduct": adduct,
                    "Neutral.Mass": row["monoisotopic_molecular_weight"],
                    "Theoretical.m.z": theoretical_mz,
                    "PPM.Error": round(ppm_error, 4),
                })

    candidates.sort(key=lambda item: (
        0 if item["Candidate.Adduct"] in PRIMARY_ADDUCTS else 1,
        abs(item["PPM.Error"]),
        str(item["HMDB"]),
        item["Candidate.Adduct"],
    ))
    return candidates





_BLANK_MATCH = {

    "HMDB": "NA", "KEGG": "NA", "Compound.Name": "NA", "Chemical.Formula": "NA",

    "CAS.Registry.Number": "NA", "ChEBI.ID": "NA", "PubChem.CID": "NA",

    "Candidate.Adduct": "NA", "Neutral.Mass": "NA",
    "Theoretical.m.z": "NA", "PPM.Error": "NA",

}


def _mapping_rows(feature_id, mz, hmdb_df, tolerance_ppm=10.0, ion_mode="auto"):
    """Build output rows for one input using only its ID and measured m/z."""
    polarity = _infer_polarity(feature_id, ion_mode)
    numeric_mz = pd.to_numeric(pd.Series([mz]), errors="coerce").iloc[0]
    base = {"Compound": str(feature_id), "MASS": numeric_mz, "Polarity": polarity}

    if pd.isna(numeric_mz) or numeric_mz <= 0:
        return [{**_BLANK_MATCH, **base, "Match.Status": "invalid_mz",
                 "Candidate.Count": 0, "Candidate.Rank": "NA"}]

    candidates = _hmdb_candidates(
        numeric_mz, hmdb_df, tolerance_ppm=tolerance_ppm, polarity=polarity
    )
    if not candidates:
        return [{**_BLANK_MATCH, **base, "Match.Status": "no_candidate",
                 "Candidate.Count": 0, "Candidate.Rank": "NA"}]

    status = "unique_mass_candidate" if len(candidates) == 1 else "ambiguous_mass_candidates"
    return [
        {**candidate, **base, "Match.Status": status,
         "Candidate.Count": len(candidates), "Candidate.Rank": rank}
        for rank, candidate in enumerate(candidates, start=1)
    ]





def generate_mapping_from_features(feature_cols, tolerance_ppm=10, results_dir=".",
                                   dataset_name="dataset", ion_mode="auto",
                                   hmdb_cache_path=HMDB_CACHE_PATH):

    """

    Auto-generate a compound-identity mapping for a list of feature names,

    using only an m/z value embedded in each feature ID. No compound names,

    pre-existing mapping

    file, HMDB ID, or KEGG ID is used as input.

    """

    import os

    import re



    if not os.path.exists(hmdb_cache_path):

        print(f"HMDB reference database not found at {hmdb_cache_path} - cannot auto-generate mapping.", flush=True)

        return None



    print("Loading HMDB reference database for mapping auto-generation...", flush=True)

    hmdb_df = _load_hmdb_reference(hmdb_cache_path)



    mz_pattern = re.compile(r"(\d+\.\d{3,})")

    results = []

    matched_features = 0



    for feat in feature_cols:

        mz_match = mz_pattern.search(feat)
        if mz_match:
            mz = float(mz_match.group(1))
            rows = _mapping_rows(feat, mz, hmdb_df, tolerance_ppm, ion_mode)
            matched_features += int(rows[0]["Candidate.Count"] > 0)
            results.extend(rows)
        else:
            results.extend(_mapping_rows(feat, np.nan, hmdb_df, tolerance_ppm, ion_mode))



    result_df = pd.DataFrame(results)

    # NAME-BASED FALLBACK: for features with no mass match but a real embedded
    # compound name (e.g. "C18-neg_Cluster_0004: 4-hydroxystyrene"), attempt an
    # exact normalized-name lookup against HMDB's own name column.
    def normalize_name(name):
        name = str(name).lower().strip()
        name = name.replace("'", "").replace(",", "").replace(".", "").replace("-", " ")
        name = re.sub(r"\s+", " ", name)
        return name

    hmdb_df["name_norm"] = hmdb_df["name"].apply(normalize_name)
    name_lookup = hmdb_df.set_index("name_norm")
    matched_by_name = 0

    for idx, row in result_df.iterrows():
        already_matched = row.get("Candidate.Count", 0) and row.get("Candidate.Count", 0) > 0
        if already_matched:
            continue
        feat = row["Compound"]
        if ": " not in str(feat):
            continue
        extracted_name = str(feat).split(": ", 1)[1].strip()
        if extracted_name.upper() == "NA" or extracted_name == "":
            continue
        extracted_norm = normalize_name(extracted_name)
        if extracted_norm in name_lookup.index:
            match = name_lookup.loc[extracted_norm]
            if isinstance(match, pd.DataFrame):
                match = match.iloc[0]
            result_df.at[idx, "HMDB"] = match["accession"]
            result_df.at[idx, "KEGG"] = match["kegg_id"] if pd.notna(match["kegg_id"]) else ""
            result_df.at[idx, "Compound.Name"] = match["name"]
            matched_by_name += 1

    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, f"generated_mapping_{dataset_name}.csv")

    result_df.to_csv(output_path, index=False)

    total_matched = matched_features + matched_by_name
    print(f"Auto-generated mapping file saved: {output_path} "
          f"({matched_features}/{len(feature_cols)} matched by mass, {matched_by_name} matched by name, "
          f"{total_matched} total).", flush=True)

    return output_path





def generate_mapping_from_id_mass_file(id_mass_file, tolerance_ppm=10, results_dir=".",
                                       dataset_name="dataset", ion_mode="auto",
                                       hmdb_cache_path=HMDB_CACHE_PATH):

    """

    Auto-generate a compound-identity mapping from a simple, separate CSV

    containing just a feature ID column and a mass column. Used when a

    dataset's raw feature names do NOT contain an embedded mass value.

    """

    import os



    if not os.path.exists(hmdb_cache_path):

        print(f"HMDB reference database not found at {hmdb_cache_path} - cannot auto-generate mapping.", flush=True)

        return None



    print(f"Loading ID+mass file for mapping generation: {id_mass_file}", flush=True)

    id_mass_df = pd.read_csv(id_mass_file, sep=None, engine="python")

    if id_mass_df.shape[1] < 2:
        raise ValueError("ID+mass file must contain at least two columns")



    mass_col = None

    for col in id_mass_df.columns:

        if col.strip().lower() in ("mass", "m.z", "mz", "m/z"):

            mass_col = col

            break

    if mass_col is None:
        raise ValueError("Could not find an m/z column named mass, mz, m.z, or m/z")

    id_candidates = [col for col in id_mass_df.columns if col != mass_col]
    id_col = next(
        (col for col in id_candidates
         if re.search(r"^(compound|feature|metabolite|cluster)(.?id)?$",
                      col.strip().lower().replace(" ", "_"))),
        id_candidates[0],
    )



    print(f"Using '{id_col}' as feature ID column and '{mass_col}' as mass column.", flush=True)



    print("Loading HMDB reference database for mapping auto-generation...", flush=True)

    hmdb_df = _load_hmdb_reference(hmdb_cache_path)



    results = []

    matched_count = 0



    for _, row in id_mass_df.iterrows():

        mz = row[mass_col]
        rows = _mapping_rows(row[id_col], mz, hmdb_df, tolerance_ppm, ion_mode)
        matched_count += int(rows[0]["Candidate.Count"] > 0)
        results.extend(rows)



    result_df = pd.DataFrame(results)

    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, f"generated_mapping_{dataset_name}.csv")

    result_df.to_csv(output_path, index=False)



    print(f"Auto-generated mapping file saved: {output_path} "

          f"({matched_count}/{len(id_mass_df)} features have at least one exact-mass candidate).", flush=True)

    return output_path





def apply_unknown_compound_filter(feature_cols, mapping_file, results_dir=".",
                                  dataset_name="dataset", tolerance_ppm=10.0,
                                  ion_mode="auto", hmdb_cache_path=HMDB_CACHE_PATH):

    if mapping_file is None:

        # Check whether feature names contain an embedded m/z value (e.g. "Negative_515.2282_5.6026")

        # If so, we can auto-generate a mapping file against the HMDB reference. If not (e.g. opaque

        # IDs like "C18-neg_Cluster_0001"), auto-generation is not possible from the features alone.

        mz_pattern = re.compile(r"(\d+\.\d{3,})")

        features_with_mz = [f for f in feature_cols if mz_pattern.search(f)]

        mz_coverage = len(features_with_mz) / max(len(feature_cols), 1)



        def has_real_name(f):

            if ": " not in f:

                return False

            extracted = f.split(": ", 1)[1].strip()

            return extracted.upper() != "NA" and extracted != ""



        features_with_name = [f for f in feature_cols if has_real_name(f)]

        name_coverage = len(features_with_name) / max(len(feature_cols), 1)



        usable_coverage = mz_coverage + name_coverage

        if usable_coverage > 0.05:

            print(f"No mapping file provided. {mz_coverage*100:.0f}% of features contain an embedded m/z value "

                  f"and {name_coverage*100:.0f}% contain a resolvable compound name - "

                  f"attempting to auto-generate a mapping file against the HMDB reference database.", flush=True)

            generated_mapping_path = generate_mapping_from_features(
                feature_cols, tolerance_ppm=tolerance_ppm, results_dir=results_dir,
                dataset_name=dataset_name, ion_mode=ion_mode,
                hmdb_cache_path=hmdb_cache_path,
            )

            if generated_mapping_path is not None:

                mapping_file = generated_mapping_path

            else:

                print("Auto-generation failed - skipping unknown-compound filter.", flush=True)

                return feature_cols

        else:

            print("No mapping file provided and feature names do not contain embedded m/z values "

                  "(e.g. pre-identified data like MetAML, or opaque feature IDs) - "

                  "skipping unknown-compound filter. To enable this filter for opaque-ID datasets, "

                  "supply a mapping file directly.", flush=True)

            return feature_cols



    mapping_df = pd.read_csv(mapping_file, sep=None, engine="python")

    print(f"Mapping file loaded: {mapping_df.shape[0]} rows, columns: {list(mapping_df.columns)}", flush=True)



    feature_set = set(feature_cols)



    id_col = None

    for col in mapping_df.columns:

        col_clean = col.strip().lower().replace(" ", "_")

        for pattern in MAPPING_ID_PATTERNS:

            if re.search(pattern, col_clean):

                overlap = mapping_df[col].astype(str).isin(feature_set).sum()

                if overlap / max(len(feature_cols), 1) > 0.10:

                    id_col = col

                    print(f"ID column matched by name pattern: '{col}' ({overlap} overlapping values)", flush=True)

                    break

        if id_col:

            break



    if id_col is None:

        best_col, best_overlap = None, 0

        for col in mapping_df.columns:

            overlap = mapping_df[col].astype(str).isin(feature_set).sum()

            if overlap > best_overlap:

                best_col, best_overlap = col, overlap

        if best_col is not None and best_overlap / max(len(feature_cols), 1) > 0.10:

            id_col = best_col

            print(f"ID column matched by content overlap: '{id_col}' ({best_overlap} overlapping values)", flush=True)



    if id_col is None:

        print("No usable mapping ID column found (overlap too low) - skipping unknown-compound filter", flush=True)

        return feature_cols



    identity_cols = []

    for col in mapping_df.columns:

        col_clean = col.strip().lower().replace(" ", "_")

        for pattern in IDENTITY_PATTERNS:

            if re.search(pattern, col_clean):

                identity_cols.append(col)

                break



    if not identity_cols:

        print("Mapping file found but no HMDB/KEGG/Compound.Name-style identity columns detected - skipping filter", flush=True)

        return feature_cols



    print(f"Identity columns detected: {identity_cols}", flush=True)



    has_identity = mapping_df[identity_cols].notna().any(axis=1)

    known_ids = set(mapping_df.loc[has_identity, id_col].astype(str))



    resolved_features = [f for f in feature_cols if f in known_ids]

    print(f"Unknown-compound filter: {len(resolved_features)}/{len(feature_cols)} features have a resolved identity", flush=True)



    return resolved_features





def sparsity_filter(mtb, feature_cols, threshold=0.20):

    presence_frac = (mtb[feature_cols] > 0).sum(axis=0) / mtb.shape[0]

    kept = presence_frac[presence_frac >= threshold].index.tolist()

    print(f"After sparsity filter: {len(kept)} features", flush=True)

    return kept





def clr_transform(data, pseudocount=1):

    data_p1 = data + pseudocount

    geo_mean = np.exp(np.log(data_p1).mean(axis=1))

    return np.log(data_p1.div(geo_mean, axis=0))





def clr_zscore_transform(mtb, sparsity_kept):
    X_clr = clr_transform(mtb[sparsity_kept].fillna(0))
    return X_clr





def encode_labels(metadata, label_col):

    label_encoder = LabelEncoder()

    y_full = label_encoder.fit_transform(metadata[label_col])

    class_names = label_encoder.classes_

    print(f"Classes auto-detected: {list(class_names)}", flush=True)

    return y_full, class_names





def split_and_remove_outliers(X_clr, y_full):
    X_train_clr, X_test_clr, y_train, y_test = train_test_split(
        X_clr, y_full, test_size=0.20, stratify=y_full, random_state=23
    )

    # Z-score fitted on TRAINING data only, then applied to both sets (no leakage)
    train_mean = X_train_clr.mean(axis=0)
    train_std = X_train_clr.std(axis=0)
    X_train = (X_train_clr - train_mean) / train_std
    X_test = (X_test_clr - train_mean) / train_std

    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples", flush=True)

    iso = IsolationForest(contamination=0.05, random_state=23)
    keep_mask = iso.fit_predict(X_train) == 1
    X_train_clean = X_train[keep_mask].reset_index(drop=True)
    y_train_clean = y_train[keep_mask]
    print(f"Isolation Forest removed {(~keep_mask).sum()} outliers, {X_train_clean.shape[0]} remain", flush=True)

    return X_train_clean, y_train_clean, X_test, y_test





def select_features_adaptive(X_train_clean, y_train_clean, class_names):

    n_train = X_train_clean.shape[0]

    print(f"Train n={n_train} (threshold=100)", flush=True)



    if n_train < 100:

        print("SMALL-SAMPLE BRANCH: skip Wilcoxon, KW top-40 direct, apply SMOTE", flush=True)

        min_class_count = pd.Series(y_train_clean).value_counts().min()

        smote_k = min(5, min_class_count - 1) if min_class_count > 1 else 1

        smote = SMOTE(random_state=23, k_neighbors=smote_k)

        X_train_bal, y_train_bal = smote.fit_resample(X_train_clean, y_train_clean)



        n_kw = min(40, X_train_bal.shape[1])

        kw_scores = []

        for feat in X_train_bal.columns:

            groups = [X_train_bal[feat][y_train_bal == c] for c in np.unique(y_train_bal)]

            try:

                stat, p = kruskal(*groups)

            except ValueError:

                stat = 0

            kw_scores.append((feat, stat))

        kw_scores.sort(key=lambda x: x[1], reverse=True)



        X_train_sig = X_train_bal[[f for f, s in kw_scores[:n_kw]]]

        y_train_final = y_train_bal

        use_class_weight = False



    else:

        print("LARGE-SAMPLE BRANCH: keep Wilcoxon+Hochberg, no KW cap", flush=True)

        significant_features = set()

        for class_idx, class_label in enumerate(class_names):

            group_a = X_train_clean[y_train_clean == class_idx]

            group_b = X_train_clean[y_train_clean != class_idx]

            p_values = []

            for feat in X_train_clean.columns:

                try:

                    stat, p = mannwhitneyu(group_a[feat], group_b[feat], alternative="two-sided")

                except ValueError:

                    p = 1.0

                p_values.append(p)

            reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="simes-hochberg")

            significant_features.update([f for f, s in zip(X_train_clean.columns, reject) if s])



        X_train_sig = X_train_clean[sorted(significant_features)]

        y_train_final = y_train_clean

        use_class_weight = True



    print(f"Candidate features going into Boruta: {X_train_sig.shape[1]}", flush=True)

    return X_train_sig, y_train_final, use_class_weight





def tune_and_run_boruta(X_train_sig, y_train_final, use_class_weight):

    rf_class_weight = 'balanced' if use_class_weight else None

    rf_param_grid = {

        'n_estimators': [100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy'],

        'max_depth': [10, 20, 30, 40, 50], 'max_features': [0.25, 0.5, 0.75, 1.0], 'max_samples': [0.8, 0.9, 1.0],

    }

    min_class_count_final = pd.Series(y_train_final).value_counts().min()

    n_inner_folds = min(10, min_class_count_final)

    skf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=23)



    rf_grid = GridSearchCV(RandomForestClassifier(random_state=23, class_weight=rf_class_weight), rf_param_grid,

                            cv=skf_inner, scoring='roc_auc_ovr', n_jobs=-1)

    rf_grid.fit(X_train_sig, y_train_final)

    rf_best_params = rf_grid.best_params_

    print(f"Best RF params: {rf_best_params}", flush=True)



    xgb_param_grid = {

        'n_estimators': [100, 200, 300, 400], 'max_depth': [5, 10, 15, 20],

        'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.8, 0.9, 1.0],

    }

    xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=23, eval_metric='mlogloss'), xgb_param_grid,

                             cv=skf_inner, scoring='roc_auc_ovr', n_jobs=-1)

    xgb_grid.fit(X_train_sig, y_train_final)

    xgb_best_params = xgb_grid.best_params_

    print(f"Best XGBoost params: {xgb_best_params}", flush=True)



    best_rf_for_boruta = RandomForestClassifier(**rf_best_params, random_state=23, class_weight=rf_class_weight)

    boruta_selector = BorutaPy(best_rf_for_boruta, n_estimators='auto', random_state=1, max_iter=50)

    boruta_selector.fit(X_train_sig.values, y_train_final)

    boruta_selected = X_train_sig.columns[boruta_selector.support_].tolist()



    if len(boruta_selected) < 3:

        tentative = X_train_sig.columns[boruta_selector.support_weak_].tolist()

        boruta_selected = list(set(boruta_selected + tentative))



    print(f"Boruta confirmed {len(boruta_selected)} features (auto-detected pipeline, default strictness)", flush=True)
    print(f"Selected features: {boruta_selected}", flush=True)



    return boruta_selected, rf_best_params, xgb_best_params, rf_class_weight





def evaluate(model, name, X_train_final, y_train_final, X_test_final, y_test, class_names):

    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test_final)



    n_classes = len(class_names)



    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test_final)

    else:

        decision = model.decision_function(X_test_final)

        if n_classes == 2:

            decision = np.column_stack([-decision, decision])

        y_proba = softmax(decision, axis=1)



    if n_classes == 2:

        auc = roc_auc_score(y_test, y_proba[:, 1])

    else:

        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")



    f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))



    sens, spec = [], []

    for i in range(len(class_names)):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - tp - fn - fp

        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)



    print(f"{name:20s} AUC={auc:.4f} Sens={np.mean(sens):.4f} Spec={np.mean(spec):.4f} F1={f1:.4f}", flush=True)

    return {"model": name, "auc": auc, "sens": np.mean(sens), "spec": np.mean(spec), "f1": f1}





def main():

    args = parse_args()



    mtb, metadata = load_data(args.features_file, args.metadata_file, args.dataset_name)

    sample_col, label_col, meta_sample_col = detect_sample_and_label_columns(mtb, metadata)



    feature_cols = [c for c in mtb.columns if c != sample_col]

    print(f"Using {len(feature_cols)} feature columns (auto-detected, excluding '{sample_col}')", flush=True)



    mapping_file_to_use = args.mapping_file

    if mapping_file_to_use is None and args.id_mass_file is not None:

        print(f"No mapping file provided, but an ID+mass file was given ({args.id_mass_file}) - "

              f"generating a compound-identity mapping from it directly.", flush=True)

        mapping_file_to_use = generate_mapping_from_id_mass_file(
            args.id_mass_file,
            tolerance_ppm=args.mass_tolerance_ppm,
            results_dir=args.results_dir,
            dataset_name=args.dataset_name,
            ion_mode=args.ion_mode,
            hmdb_cache_path=args.hmdb_cache,
        )

    feature_cols = apply_unknown_compound_filter(
        feature_cols,
        mapping_file_to_use,
        results_dir=args.results_dir,
        dataset_name=args.dataset_name,
        tolerance_ppm=args.mass_tolerance_ppm,
        ion_mode=args.ion_mode,
        hmdb_cache_path=args.hmdb_cache,
    )



    y_full, class_names = encode_labels(metadata, label_col)

    # Split RAW samples into train/test FIRST, before any filtering or transformation (no leakage)
    train_idx, test_idx = train_test_split(
        np.arange(mtb.shape[0]), test_size=0.20, stratify=y_full, random_state=23
    )
    mtb_train = mtb.iloc[train_idx].reset_index(drop=True)
    mtb_test = mtb.iloc[test_idx].reset_index(drop=True)
    y_train_raw = y_full[train_idx]
    y_test = y_full[test_idx]

    sparsity_kept = sparsity_filter(mtb_train, feature_cols)

    X_train_clr = clr_transform(mtb_train[sparsity_kept].fillna(0))
    X_test_clr = clr_transform(mtb_test[sparsity_kept].fillna(0))

    train_mean = X_train_clr.mean(axis=0)
    train_std = X_train_clr.std(axis=0)
    X_train_z = (X_train_clr - train_mean) / train_std
    X_test = (X_test_clr - train_mean) / train_std

    iso = IsolationForest(contamination=0.05, random_state=23)
    keep_mask = iso.fit_predict(X_train_z) == 1
    X_train_clean = X_train_z[keep_mask].reset_index(drop=True)
    y_train_clean = y_train_raw[keep_mask]
    print(f"Isolation Forest removed {(~keep_mask).sum()} outliers from TRAIN only, {X_train_clean.shape[0]} remain", flush=True)

    X_train_sig, y_train_final, use_class_weight = select_features_adaptive(X_train_clean, y_train_clean, class_names)

    boruta_selected, rf_best_params, xgb_best_params, rf_class_weight = tune_and_run_boruta(

        X_train_sig, y_train_final, use_class_weight

    )



    X_train_final = X_train_sig[boruta_selected]

    X_test_final = X_test[boruta_selected]



    print("\n" + "=" * 70, flush=True)

    print(f"UNIVERSAL PIPELINE - FULL RUN ON {args.dataset_name}", flush=True)

    print("=" * 70, flush=True)



    lasso_class_weight = 'balanced' if use_class_weight else None



    results = []

    results.append(evaluate(

        RandomForestClassifier(**rf_best_params, random_state=23, class_weight=rf_class_weight),

        "RandomForest", X_train_final, y_train_final, X_test_final, y_test, class_names

    ))

    results.append(evaluate(

        xgb.XGBClassifier(**xgb_best_params, random_state=23, eval_metric='mlogloss'),

        "XGBoost", X_train_final, y_train_final, X_test_final, y_test, class_names

    ))

    results.append(evaluate(

        RidgeClassifier(random_state=23, class_weight=rf_class_weight),

        "Ridge", X_train_final, y_train_final, X_test_final, y_test, class_names

    ))

    results.append(evaluate(

        LogisticRegression(penalty='l1', solver='saga', max_iter=5000, class_weight=lasso_class_weight, random_state=23),

        "LASSO", X_train_final, y_train_final, X_test_final, y_test, class_names

    ))



    best_result = max(results, key=lambda r: r["auc"])

    print(f"\nBEST MODEL: {best_result['model']} with AUC={best_result['auc']:.4f}", flush=True)

    print(f"\n*** SUCCESS: Universal pipeline ran end-to-end on {args.dataset_name} ***", flush=True)



    out_path = f"{args.results_dir}/{args.dataset_name}_universal_results.csv"

    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"Saved {out_path}", flush=True)





if __name__ == "__main__":

    main()
