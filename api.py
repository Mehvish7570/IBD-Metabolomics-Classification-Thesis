from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import subprocess
import tempfile
import shutil
import os
from sklearn.metrics import roc_auc_score, confusion_matrix

app = FastAPI(title="MetaboIBD API")

# ---- Load all datasets' model bundles once at startup ----
DATASETS = {
    "franzosa": joblib.load("models/all_models.pkl"),
    "jacobs": joblib.load("models/jacobs_models.pkl"),
    "metaml": joblib.load("models/metaml_models.pkl"),
}

# ---- Published AUC values, from each paper's own reported results ----
# IMPORTANT: verify these against your literature table before using in the thesis —
# these are pulled from your own notes, not re-checked against the papers in this session.
# Also note: the MetAML value below is from MetAML's OWN dataset (Pasolli et al. 2016),
# not Franzosa — it is not a like-for-like comparison, only included for completeness.
# There is no published baseline for Jacobs or for approaches run on the MetAML dataset
# itself (not the original paper's dataset for any of these methods), so those entries
# will show published_auc = null.
PUBLISHED_AUC = {
    "franzosa": {
        "acharjee": 0.953,   # Hodgkiss & Acharjee 2025, XGBoost, Franzosa
        "ibdpred": 0.852,    # Linares Blanco et al. 2022, Elastic Net, Franzosa
        "siamcat": 0.919,    # Wirbel et al. 2021 (SIAMCAT applied to Franzosa), Random Forest
        "vichvila": 0.925,   # Vich Vila et al. 2023
        "metaml": 0.890,     # Pasolli et al. 2016, Random Forest — on MetAML's own dataset, NOT Franzosa
    },
    "jacobs": {},
    "metaml": {},
}

METHODOLOGY_INFO = {
    "acharjee": "Sparsity filter + CLR + Z-score (Hodgkiss and Acharjee 2025)",
    "ibdpred": "Kruskal-Wallis feature ranking (Linares Blanco et al. 2022)",
    "siamcat": "AUC-based feature ranking (Wirbel et al. 2021)",
    "vichvila": "CLR + LASSO (Vich Vila et al. 2023)",
    "metaml": "Zero-variance filter only, minimal preprocessing (Pasolli et al. 2016)",
}

# ---- Path to the universal pipeline script (must be in the same folder as this file) ----
UNIVERSAL_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "own_method_universal_LEAKAGE_CORRECTED_final.py")

# ---- Path to the missingness/correlation/PCA/ranked-features data (per dataset) ----
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis_data")


@app.get("/")
def root():
    return {
        "message": "MetaboIBD API is running",
        "datasets": list(DATASETS.keys()),
    }


@app.get("/datasets")
def list_datasets():
    """List available datasets and which methodologies exist for each."""
    return {
        name: {
            "methodologies": list(models.keys()),
            "test_set_size": len(next(iter(models.values()))["y_test"]),
        }
        for name, models in DATASETS.items()
    }


@app.get("/compare")
def compare_all(dataset: str = "franzosa"):
    """
    Compare all methodologies on the given dataset's shared held-out test set.
    Returns, per methodology: reproduced AUC, published AUC (if one exists),
    confusion matrix, test set size, and (for ROC curves) the true labels and
    predicted probabilities.
    """
    if dataset not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}",
        )
    models = DATASETS[dataset]
    published = PUBLISHED_AUC.get(dataset, {})
    results = []
    for method, entry in models.items():
        classifier_key = "model_xgb" if "model_xgb" in entry else "model"
        model = entry[classifier_key]
        X_test = entry["X_test"]
        y_test = entry["y_test"]
        probs = model.predict_proba(X_test)
        if probs.shape[1] == 2:
            reproduced_auc = roc_auc_score(y_test, probs[:, 1])
        else:
            reproduced_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds).tolist()
        results.append({
            "methodology": method,
            "description": METHODOLOGY_INFO.get(method, ""),
            "reproduced_auc": round(float(reproduced_auc), 4),
            "published_auc": published.get(method),
            "confusion_matrix": cm,
            "test_set_size": len(y_test),
            "y_test": [int(v) for v in y_test],
            "probabilities": probs.tolist(),
        })
    results.sort(key=lambda x: -x["reproduced_auc"])
    return {"dataset": dataset, "comparison": results}


@app.get("/analysis/missingness")
def get_missingness(dataset: str = "franzosa"):
    """Return the missingness report for a dataset."""
    path = os.path.join(ANALYSIS_DIR, dataset, "missingness_report.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No missingness report found for '{dataset}'.")
    df = pd.read_csv(path)
    return {"dataset": dataset, "missingness": df.to_dict(orient="records")}


@app.get("/analysis/correlation")
def get_correlation(dataset: str = "franzosa"):
    """Return the top correlated feature pairs for a dataset."""
    path = os.path.join(ANALYSIS_DIR, dataset, "correlation_top_pairs.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No correlation report found for '{dataset}'.")
    df = pd.read_csv(path)
    return {"dataset": dataset, "correlations": df.to_dict(orient="records")}


@app.get("/analysis/ranked-features")
def get_ranked_features(dataset: str = "franzosa", approach: str = "acharjee"):
    """Return the ranked/important features for a dataset+approach combination."""
    path = os.path.join(ANALYSIS_DIR, dataset, f"ranked_features_{approach}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No ranked-features file found for '{dataset}'/'{approach}'.")
    df = pd.read_csv(path)
    return {"dataset": dataset, "approach": approach, "ranked_features": df.head(20).to_dict(orient="records")}


@app.get("/analysis/pca-plot")
def get_pca_plot(dataset: str = "franzosa"):
    """Return the PCA plot image for a dataset."""
    path = os.path.join(ANALYSIS_DIR, dataset, "pca_plot.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No PCA plot found for '{dataset}'.")
    return FileResponse(path, media_type="image/png")


@app.post("/universal/analyze")
async def universal_analyze(
    features_file: UploadFile = File(...),
    metadata_file: UploadFile = File(...),
    mapping_file: UploadFile = File(None),
):
    """
    Run the universal pipeline on an uploaded dataset (any disease, any feature
    naming convention, with or without a compound-mapping file). Trains and
    tests a fresh model end-to-end using the same fixed methodology as the
    thesis's own combined approach: sparsity filter, optional unknown-compound
    filter, CLR + Z-score, 80/20 split, Isolation Forest, adaptive feature
    selection, Boruta, grid-searched RF/XGBoost, and a 4-classifier comparison.
    """
    if not os.path.exists(UNIVERSAL_SCRIPT_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Universal pipeline script not found at {UNIVERSAL_SCRIPT_PATH}",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, features_file.filename)
        metadata_path = os.path.join(tmpdir, metadata_file.filename)

        with open(features_path, "wb") as f:
            shutil.copyfileobj(features_file.file, f)
        with open(metadata_path, "wb") as f:
            shutil.copyfileobj(metadata_file.file, f)

        mapping_path = None
        if mapping_file is not None:
            mapping_path = os.path.join(tmpdir, mapping_file.filename)
            with open(mapping_path, "wb") as f:
                shutil.copyfileobj(mapping_file.file, f)

        cmd = [
            "python", UNIVERSAL_SCRIPT_PATH,
            "--features_file", features_path,
            "--metadata_file", metadata_path,
            "--dataset_name", "UserUpload",
            "--results_dir", tmpdir,
        ]
        if mapping_path:
            cmd += ["--mapping_file", mapping_path]

        env = os.environ.copy()
        env["SCIPY_ARRAY_API"] = "1"

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, env=env
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Analysis timed out after 10 minutes.")

        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline failed: {result.stderr[-2000:]}",
            )

        results_csv = os.path.join(tmpdir, "UserUpload_universal_results.csv")
        if not os.path.exists(results_csv):
            raise HTTPException(
                status_code=500,
                detail="Pipeline completed but no results file was produced.",
            )

        results_df = pd.read_csv(results_csv)
        best_row = results_df.loc[results_df["auc"].idxmax()]

        generated_mapping = None
        generated_mapping_path = "generated_mapping_file.tsv"
        if os.path.exists(generated_mapping_path):
            mapping_df = pd.read_csv(generated_mapping_path, sep="\t")
            matched_df = mapping_df[mapping_df["HMDB"].notna() & (mapping_df["HMDB"] != "")].copy()
            matched_df = matched_df.fillna("")
            generated_mapping = {
                "total_features": len(mapping_df),
                "matched_features": len(matched_df),
                "preview": matched_df.head(20).to_dict(orient="records"),
            }

        return {
            "log": result.stdout,
            "results": results_df.to_dict(orient="records"),
            "best_model": best_row["model"],
            "best_auc": round(float(best_row["auc"]), 4),
            "generated_mapping": generated_mapping,
        }