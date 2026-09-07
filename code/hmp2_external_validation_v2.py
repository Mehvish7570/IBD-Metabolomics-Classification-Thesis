
# ============================================================================

# HMP2 EXTERNAL VALIDATION v2 - FIXED to match by chemical compound name,

# not raw cluster ID (which never transfers across independently-processed

# studies). Mirrors Acharjee/Hodgkiss's own real validation approach.

# Trains on 100% of Franzosa, tests on 100% of HMP2, no data crossover.

# ============================================================================



import pandas as pd

import numpy as np

from scipy.stats import mannwhitneyu

from scipy.special import softmax

from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from boruta import BorutaPy

import xgboost as xgb



RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"

FRANZOSA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

HMP2_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/iHMP_IBDMDB_2019"



# ---------------- TRAIN on 100% of Franzosa, NAMED COMPOUNDS ONLY ----------------

mtb = pd.read_csv(f"{FRANZOSA_DIR}/mtb.tsv", sep="\t")

metadata = pd.read_csv(f"{FRANZOSA_DIR}/metadata.tsv", sep="\t")

mtb_map = pd.read_csv(f"{FRANZOSA_DIR}/mtb.map.tsv", sep="\t")

feature_cols = [c for c in mtb.columns if c != "Sample"]

print(f"Loaded Franzosa (TRAIN): {mtb.shape[0]} samples, {len(feature_cols)} raw features", flush=True)



presence_frac = (mtb[feature_cols] > 0).sum(axis=0) / mtb.shape[0]

sparsity_kept = presence_frac[presence_frac >= 0.20].index.tolist()

print(f"After sparsity filter: {len(sparsity_kept)} features", flush=True)



# NEW: restrict to features with a RESOLVED compound name (required for

# cross-study matching by chemical identity - matches Acharjee's approach)

named_map = mtb_map[mtb_map["Compound.Name"].notna()][["Compound", "Compound.Name"]].copy()

named_map["Compound.Name.Norm"] = named_map["Compound.Name"].str.strip().str.lower()

compound_to_name = dict(zip(named_map["Compound"], named_map["Compound.Name.Norm"]))

own_features = [f for f in sparsity_kept if f in compound_to_name]

print(f"After restricting to NAMED compounds only: {len(own_features)} features", flush=True)



def clr_transform(data, pseudocount=1):

    data_p1 = data + pseudocount

    geo_mean = np.exp(np.log(data_p1).mean(axis=1))

    return np.log(data_p1.div(geo_mean, axis=0))



X_clr = clr_transform(mtb[own_features].fillna(0))

X_zscore = (X_clr - X_clr.mean(axis=0)) / X_clr.std(axis=0)



label_encoder = LabelEncoder()

y_full = label_encoder.fit_transform(metadata["Study.Group"])

class_names = label_encoder.classes_

print(f"Franzosa classes: {list(class_names)}", flush=True)



iso = IsolationForest(contamination=0.05, random_state=23)

outlier_pred = iso.fit_predict(X_zscore)

keep_mask = outlier_pred == 1

X_train_clean = X_zscore[keep_mask].reset_index(drop=True)

y_train_clean = y_full[keep_mask]

print(f"Isolation Forest removed {(~keep_mask).sum()} outliers, {X_train_clean.shape[0]} remain", flush=True)



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

    sig_this_class = [feat for feat, is_sig in zip(X_train_clean.columns, reject) if is_sig]

    significant_features.update(sig_this_class)

    print(f"{class_label} vs rest: {len(sig_this_class)} significant features", flush=True)

significant_features = sorted(significant_features)

print(f"After Wilcoxon+Hochberg: {len(significant_features)} features", flush=True)

X_train_sig = X_train_clean[significant_features]



if X_train_sig.shape[1] < 3:

    print("ERROR: too few named+significant features to proceed. Stopping.", flush=True)

    raise SystemExit(1)



param_grid = {

    'n_estimators': [100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy'],

    'max_depth': [10, 20, 30, 40, 50], 'max_features': [0.25, 0.5, 0.75, 1.0], 'max_samples': [0.8, 0.9, 1.0],

}

skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)



rf_grid = GridSearchCV(RandomForestClassifier(random_state=23, class_weight='balanced'), param_grid,

                        cv=skf_inner, scoring='roc_auc_ovr', n_jobs=-1)

rf_grid.fit(X_train_sig, y_train_clean)

rf_best_params = rf_grid.best_params_

print(f"Best RF params: {rf_best_params}", flush=True)



xgb_param_grid = {

    'n_estimators': [100, 200, 300, 400], 'max_depth': [5, 10, 15, 20],

    'learning_rate': [0.01, 0.05, 0.1, 0.2], 'subsample': [0.8, 0.9, 1.0],

}

xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=23, eval_metric='mlogloss'), xgb_param_grid,

                         cv=skf_inner, scoring='roc_auc_ovr', n_jobs=-1)

xgb_grid.fit(X_train_sig, y_train_clean)

xgb_best_params = xgb_grid.best_params_

print(f"Best XGBoost params: {xgb_best_params}", flush=True)



best_rf_for_boruta = RandomForestClassifier(**rf_best_params, random_state=23, class_weight='balanced')

boruta_selector = BorutaPy(best_rf_for_boruta, n_estimators='auto', random_state=1, max_iter=50)

boruta_selector.fit(X_train_sig.values, y_train_clean)

boruta_selected = X_train_sig.columns[boruta_selector.support_].tolist()

if len(boruta_selected) < 3:

    tentative = X_train_sig.columns[boruta_selector.support_weak_].tolist()

    boruta_selected = list(set(boruta_selected + tentative))

print(f"Boruta confirmed {len(boruta_selected)} NAMED features", flush=True)

for f in boruta_selected:

    print(f"  {f} -> {compound_to_name.get(f, 'UNKNOWN')}", flush=True)



X_train_final = X_train_sig[boruta_selected]



models = {

    "RandomForest": RandomForestClassifier(**rf_best_params, random_state=23, class_weight='balanced'),

    "XGBoost": xgb.XGBClassifier(**xgb_best_params, random_state=23, eval_metric='mlogloss'),

    "Ridge": RidgeClassifier(random_state=23, class_weight='balanced'),

    "LASSO": LogisticRegression(penalty='l1', solver='saga', max_iter=5000, class_weight='balanced', random_state=23),

}

for name, model in models.items():

    model.fit(X_train_final, y_train_clean)

print("All 4 models trained on full Franzosa (named compounds only).", flush=True)



# ---------------- TEST on 100% of HMP2, matched by COMPOUND NAME ----------------

hmp2_mtb = pd.read_csv(f"{HMP2_DIR}/mtb.tsv/mtb.tsv", sep="\t")

hmp2_metadata = pd.read_csv(f"{HMP2_DIR}/metadata.tsv", sep="\t")

hmp2_map = pd.read_csv(f"{HMP2_DIR}/mtb.map.tsv", sep="\t")

print(f"\nLoaded HMP2 (TEST): {hmp2_mtb.shape[0]} samples", flush=True)



label_map = {}

for val in hmp2_metadata["Study.Group"].unique():

    val_lower = str(val).lower()

    if "cd" in val_lower or "crohn" in val_lower:

        label_map[val] = "CD"

    elif "uc" in val_lower or "colitis" in val_lower:

        label_map[val] = "UC"

    else:

        label_map[val] = "Control"

print(f"Label mapping applied: {label_map}", flush=True)

hmp2_mapped_labels = hmp2_metadata["Study.Group"].map(label_map)

y_hmp2 = label_encoder.transform(hmp2_mapped_labels)



# Build HMP2 compound-name -> HMP2 feature-ID lookup (normalized string match)

hmp2_named_map = hmp2_map[hmp2_map["Compound.Name"].notna()][["Compound", "Compound.Name"]].copy()

hmp2_named_map["Compound.Name.Norm"] = hmp2_named_map["Compound.Name"].str.strip().str.lower()

name_to_hmp2_id = {}

for _, row in hmp2_named_map.iterrows():

    name_to_hmp2_id.setdefault(row["Compound.Name.Norm"], row["Compound"])



hmp2_feature_cols = [c for c in hmp2_mtb.columns if c != "Sample"]

X_hmp2_raw = hmp2_mtb[hmp2_feature_cols].fillna(0)



# Map each Franzosa Boruta-selected feature to its HMP2 equivalent by name

matched_pairs = []

for franzosa_feat in boruta_selected:

    compound_name = compound_to_name.get(franzosa_feat)

    if compound_name and compound_name in name_to_hmp2_id:

        hmp2_feat = name_to_hmp2_id[compound_name]

        if hmp2_feat in X_hmp2_raw.columns:

            matched_pairs.append((franzosa_feat, hmp2_feat, compound_name))



print(f"\nMatched {len(matched_pairs)}/{len(boruta_selected)} Boruta features to HMP2 by compound name:", flush=True)

for f, h, n in matched_pairs:

    print(f"  {f} <-> {h} ({n})", flush=True)



if len(matched_pairs) < 3:

    print(f"ERROR: only {len(matched_pairs)} features matched - too few for reliable evaluation. Stopping.", flush=True)

    raise SystemExit(1)



franzosa_matched_feats = [p[0] for p in matched_pairs]

hmp2_matched_feats = [p[1] for p in matched_pairs]



X_hmp2_clr = clr_transform(X_hmp2_raw[hmp2_matched_feats])

X_hmp2_zscore = (X_hmp2_clr - X_hmp2_clr.mean(axis=0)) / X_hmp2_clr.std(axis=0)

X_hmp2_zscore.columns = franzosa_matched_feats  # rename to match Franzosa's feature names

X_hmp2_final = X_hmp2_zscore.fillna(0)



X_train_final_matched = X_train_final[franzosa_matched_feats]

for name, model in models.items():

    model.fit(X_train_final_matched, y_train_clean)



print("\n" + "=" * 70, flush=True)

print("HMP2 EXTERNAL VALIDATION v2 RESULTS (matched by compound name)", flush=True)

print("=" * 70, flush=True)



results = []

for name, model in models.items():

    y_pred = model.predict(X_hmp2_final)

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_hmp2_final)

    else:

        y_proba = softmax(model.decision_function(X_hmp2_final), axis=1)

    present_classes = np.unique(y_hmp2)

    if len(present_classes) < len(class_names):

        print(f"{name:20s} SKIPPED - HMP2 test set missing a class present in training", flush=True)

        continue

    auc = roc_auc_score(y_hmp2, y_proba, multi_class="ovr", average="macro")

    f1 = f1_score(y_hmp2, y_pred, average="macro")

    cm = confusion_matrix(y_hmp2, y_pred, labels=range(len(class_names)))

    sens, spec = [], []

    for i in range(len(class_names)):

        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp

        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    print(f"{name:20s} AUC={auc:.4f} Sens={np.mean(sens):.4f} Spec={np.mean(spec):.4f} F1={f1:.4f}", flush=True)

    results.append({"model": name, "auc": auc, "sens": np.mean(sens), "spec": np.mean(spec), "f1": f1})



if results:

    best_result = max(results, key=lambda r: r["auc"])

    print(f"\nBEST MODEL: {best_result['model']} with AUC={best_result['auc']:.4f}", flush=True)

    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/hmp2_external_validation_v2_results.csv", index=False)

    print("Saved hmp2_external_validation_v2_results.csv", flush=True)

print("Done.", flush=True)

