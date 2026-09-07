
# ============================================================================

# FINAL COMBINED METHODOLOGY - Franzosa

# Synthesized from Styles 1-10: leakage-fixed 80/20 split, adaptive rule on

# train sample size (n>=100 here so keeps Wilcoxon+Hochberg, no KW cap),

# Boruta default strictness, BOTH RF and XGBoost grid-searched, best of 4

# classifiers reported.

# ============================================================================



import pandas as pd

import numpy as np

from scipy.stats import mannwhitneyu, kruskal, ttest_ind

from scipy.special import softmax

from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from boruta import BorutaPy

from imblearn.over_sampling import SMOTE

import xgboost as xgb



RESULTS_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/results"

DATA_DIR = "/rds/projects/g/guellili-slr-automation/mehvish/thesis/data/FRANZOSA_IBD_2019"

DATASET_NAME = "Franzosa"



mtb = pd.read_csv(f"{DATA_DIR}/mtb.tsv", sep="\t")

metadata = pd.read_csv(f"{DATA_DIR}/metadata.tsv", sep="\t")

mtb_map = pd.read_csv(f"{DATA_DIR}/mtb.map.tsv", sep="\t")

feature_cols = [c for c in mtb.columns if c != "Sample"]

print(f"Loaded {DATASET_NAME}: {mtb.shape[0]} samples, {len(feature_cols)} raw features", flush=True)



presence_frac = (mtb[feature_cols] > 0).sum(axis=0) / mtb.shape[0]

sparsity_kept = presence_frac[presence_frac >= 0.20].index.tolist()

identified_features = mtb_map[

    mtb_map["HMDB"].notna() | mtb_map["KEGG"].notna() |

    mtb_map["Compound.Name"].notna() | mtb_map["Putative.Chemical.Class"].notna()

]["Compound"].tolist()

own_features = [f for f in sparsity_kept if f in identified_features]

print(f"After sparsity+unknown-compound filters: {len(own_features)} features", flush=True)



def clr_transform(data, pseudocount=1):

    data_p1 = data + pseudocount

    geo_mean = np.exp(np.log(data_p1).mean(axis=1))

    return np.log(data_p1.div(geo_mean, axis=0))



X_clr = clr_transform(mtb[own_features].fillna(0))

X_zscore = (X_clr - X_clr.mean(axis=0)) / X_clr.std(axis=0)



label_encoder = LabelEncoder()

y_full = label_encoder.fit_transform(metadata["Study.Group"])

class_names = label_encoder.classes_

print(f"Classes: {list(class_names)}", flush=True)



X_train, X_test, y_train, y_test = train_test_split(

    X_zscore, y_full, test_size=0.20, stratify=y_full, random_state=23

)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples", flush=True)



iso = IsolationForest(contamination=0.05, random_state=23)

outlier_pred = iso.fit_predict(X_train)

keep_mask = outlier_pred == 1

X_train_clean = X_train[keep_mask].reset_index(drop=True)

y_train_clean = y_train[keep_mask]

print(f"Isolation Forest removed {(~keep_mask).sum()} outliers from TRAIN only, {X_train_clean.shape[0]} remain", flush=True)



# ADAPTIVE RULE: branch on train sample size (n>=100 here)

SMALL_SAMPLE_THRESHOLD = 100

n_train = X_train_clean.shape[0]

print(f"Train n={n_train}, threshold={SMALL_SAMPLE_THRESHOLD}", flush=True)



if n_train < SMALL_SAMPLE_THRESHOLD:

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

    kw_top = [f for f, s in kw_scores[:n_kw]]

    X_train_sig = X_train_bal[kw_top]

    y_train_final = y_train_bal

    use_class_weight = False

else:

    print("LARGE-SAMPLE BRANCH: keep Wilcoxon+Hochberg, no KW cap", flush=True)

    mummichog_pvals = {}

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

        for feat, p in zip(X_train_clean.columns, p_adj):
            if feat not in mummichog_pvals or p < mummichog_pvals[feat]:
                mummichog_pvals[feat] = p

        print(f"{class_label} vs rest: {len(sig_this_class)} significant features", flush=True)

    significant_features = sorted(significant_features)

    # Compute CD-vs-Control and UC-vs-Control t-scores for Mummichog (strongest signal per feature)
    mummichog_tscores = {}
    cd_idx = list(class_names).index("CD")
    uc_idx = list(class_names).index("UC")
    control_idx = list(class_names).index("Control")
    control_vals_df = X_train_clean[y_train_clean == control_idx]
    cd_vals_df = X_train_clean[y_train_clean == cd_idx]
    uc_vals_df = X_train_clean[y_train_clean == uc_idx]
    for feat in X_train_clean.columns:
        try:
            t_cd, _ = ttest_ind(cd_vals_df[feat], control_vals_df[feat], equal_var=False)
        except ValueError:
            t_cd = 0.0
        try:
            t_uc, _ = ttest_ind(uc_vals_df[feat], control_vals_df[feat], equal_var=False)
        except ValueError:
            t_uc = 0.0
        mummichog_tscores[feat] = t_cd if abs(t_cd) >= abs(t_uc) else t_uc

    print(f"After Wilcoxon+Hochberg: {len(significant_features)} features", flush=True)

    X_train_sig = X_train_clean[significant_features]

    y_train_final = y_train_clean

    use_class_weight = True



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

print(f"Boruta confirmed {len(boruta_selected)} features (default strictness)", flush=True)



X_train_final = X_train_sig[boruta_selected]

X_test_final = X_test[boruta_selected]



# Save data for Visual 4 (boxplots) and Visual 5 (heatmap)

import joblib

feature_data = {

    "X_train_final": X_train_final,

    "y_train_final": y_train_final,

    "X_test_final": X_test_final,

    "y_test": y_test,

    "class_names": list(class_names),

    "boruta_selected": boruta_selected,

    "mtb_map": mtb_map,

}

joblib.dump(feature_data, "feature_data_franzosa.pkl")

print("Saved feature_data_franzosa.pkl", flush=True)

# Save data for Mummichog pathway analysis (m/z, retention time, p-value per feature)
mummichog_df = mtb_map[mtb_map["Compound"].isin(mummichog_pvals.keys())].copy()

mummichog_df["p_value"] = mummichog_df["Compound"].map(mummichog_pvals)

mummichog_df["t_score"] = mummichog_df["Compound"].map(mummichog_tscores)

mummichog_df = mummichog_df[["Compound", "m.z", "Retention.Time", "p_value", "t_score"]]

mummichog_df.to_csv("mummichog_input_franzosa.csv", index=False)

print(f"Saved mummichog_input_franzosa.csv ({len(mummichog_df)} features)", flush=True)




roc_data = {}

def evaluate(model, name):

    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test_final)

    if hasattr(model, "predict_proba"):

        y_proba = model.predict_proba(X_test_final)

    else:

        y_proba = softmax(model.decision_function(X_test_final), axis=1)

    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    roc_data[name] = {"y_test": y_test, "y_proba": y_proba, "class_names": list(class_names)}

    f1 = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

    sens, spec = [], []

    for i in range(len(class_names)):

        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp

        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    print(f"{name:20s} AUC={auc:.4f} Sens={np.mean(sens):.4f} Spec={np.mean(spec):.4f} F1={f1:.4f}", flush=True)

    return {"model": name, "auc": auc, "sens": np.mean(sens), "spec": np.mean(spec), "f1": f1}



print("\n" + "=" * 70, flush=True)

print(f"FINAL COMBINED METHODOLOGY - {DATASET_NAME}", flush=True)

print("=" * 70, flush=True)



lasso_class_weight = 'balanced' if use_class_weight else None

results = []

results.append(evaluate(RandomForestClassifier(**rf_best_params, random_state=23, class_weight=rf_class_weight), "RandomForest"))

results.append(evaluate(xgb.XGBClassifier(**xgb_best_params, random_state=23, eval_metric='mlogloss'), "XGBoost"))

results.append(evaluate(RidgeClassifier(random_state=23, class_weight=rf_class_weight), "Ridge"))

results.append(evaluate(LogisticRegression(penalty='l1', solver='saga', max_iter=5000, class_weight=lasso_class_weight, random_state=23), "LASSO"))



best_result = max(results, key=lambda r: r["auc"])

print(f"\nBEST MODEL: {best_result['model']} with AUC={best_result['auc']:.4f}", flush=True)



pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/own_method_final_{DATASET_NAME.lower()}_all_models.csv", index=False)

pd.DataFrame([{

    "Methodology": "Own Combined Method (FINAL)", "Dataset": DATASET_NAME,

    "Best_Model": best_result["model"], "AUC": round(best_result["auc"], 4),

    "Sensitivity": round(best_result["sens"], 4), "Specificity": round(best_result["spec"], 4),

    "F1_score": round(best_result["f1"], 4), "N_train": X_train_final.shape[0],

    "N_test": X_test_final.shape[0], "N_features_selected": len(boruta_selected),

}]).to_csv(f"{RESULTS_DIR}/own_method_final_{DATASET_NAME.lower()}_BEST.csv", index=False)

print("Saved results.", flush=True)




import joblib

joblib.dump(roc_data, "roc_data_franzosa.pkl")

print("Saved roc_data_franzosa.pkl", flush=True)

