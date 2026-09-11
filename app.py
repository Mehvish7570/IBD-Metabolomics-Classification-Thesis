import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

st.set_page_config(page_title="IBD Metabolomics Classifier", layout="wide")
st.title("MetaboIBD — IBD Metabolomics Classification Tool")

API_URL = "http://127.0.0.1:8000"

franzosa_models = joblib.load("models/all_models.pkl")
jacobs_models = joblib.load("models/jacobs_models.pkl")
metaml_models = joblib.load("models/metaml_models.pkl")
DATASETS = {"Franzosa": franzosa_models, "Jacobs": jacobs_models, "MetAML": metaml_models}

approach_info = {
    "acharjee": "Sparsity filter + CLR + Z-score (Hodgkiss and Acharjee 2025)",
    "ibdpred": "Kruskal-Wallis feature ranking (Linares Blanco et al. 2022)",
    "siamcat": "AUC-based feature ranking (Wirbel et al. 2021)",
    "vichvila": "CLR + LASSO (Vich Vila et al. 2023)",
    "metaml": "Zero-variance filter only, minimal preprocessing (Pasolli et al. 2016)",
}

CLASSIFIER_LABELS = {
    "model": "Random Forest",
    "model_xgb": "XGBoost",
    "model_enet": "Elastic Net",
    "model_lasso": "LASSO",
    "model_ridge": "Ridge",
}

# ---- Task 3 main table (80/20 split, our own) + Task 2 published AUC, finalized ----
MAIN_TABLE = [
    {"Approach": "Acharjee/Hodgkiss", "Dataset": "Franzosa", "AUC": 0.879, "Sensitivity": 0.757, "Specificity": 0.872, "Published AUC": "0.94"},
    {"Approach": "Acharjee/Hodgkiss", "Dataset": "Jacobs",   "AUC": 0.730, "Sensitivity": 0.406, "Specificity": 0.710, "Published AUC": "—"},
    {"Approach": "Acharjee/Hodgkiss", "Dataset": "MetAML",   "AUC": 0.942, "Sensitivity": 0.740, "Specificity": 0.880, "Published AUC": "—"},
    {"Approach": "IBDpred", "Dataset": "Franzosa", "AUC": 0.819, "Sensitivity": 0.621, "Specificity": 0.807, "Published AUC": "0.76 / 0.74"},
    {"Approach": "IBDpred", "Dataset": "Jacobs",   "AUC": 0.832, "Sensitivity": 0.567, "Specificity": 0.855, "Published AUC": "—"},
    {"Approach": "IBDpred", "Dataset": "MetAML",   "AUC": 0.959, "Sensitivity": 0.896, "Specificity": 0.911, "Published AUC": "—"},
    {"Approach": "SIAMCAT", "Dataset": "Franzosa", "AUC": 0.885, "Sensitivity": 0.750, "Specificity": 0.868, "Published AUC": "0.76 / 0.84"},
    {"Approach": "SIAMCAT", "Dataset": "Jacobs",   "AUC": 0.514, "Sensitivity": 0.369, "Specificity": 0.688, "Published AUC": "—"},
    {"Approach": "SIAMCAT", "Dataset": "MetAML",   "AUC": 0.937, "Sensitivity": 0.825, "Specificity": 0.897, "Published AUC": "—"},
    {"Approach": "Vich Vila", "Dataset": "Franzosa", "AUC": 0.827, "Sensitivity": 0.665, "Specificity": 0.824, "Published AUC": "0.85 / 0.83"},
    {"Approach": "Vich Vila", "Dataset": "Jacobs",   "AUC": 0.689, "Sensitivity": 0.442, "Specificity": 0.747, "Published AUC": "—"},
    {"Approach": "Vich Vila", "Dataset": "MetAML",   "AUC": 0.909, "Sensitivity": 0.701, "Specificity": 0.901, "Published AUC": "—"},
    {"Approach": "MetAML", "Dataset": "Franzosa", "AUC": 0.890, "Sensitivity": 0.710, "Specificity": 0.847, "Published AUC": "—"},
    {"Approach": "MetAML", "Dataset": "Jacobs",   "AUC": 0.727, "Sensitivity": 0.533, "Specificity": 0.810, "Published AUC": "—"},
    {"Approach": "MetAML", "Dataset": "MetAML",   "AUC": 0.957, "Sensitivity": 0.699, "Specificity": 0.897, "Published AUC": "0.890 (CI 0.812–0.968)"},
]
MAIN_TABLE_DF = pd.DataFrame(MAIN_TABLE)

with st.expander("Read me first — what this tool does and how to use it", expanded=True):
    st.markdown("""
    **What this is:** a comparison tool for classifying gut microbiome
    metabolite profiles into Inflammatory Bowel Disease (IBD) versus healthy controls,
    using several published approaches reproduced on the same data.

    **How to use each tab:**
    - **Dataset Comparison** — see how all reproduced approaches perform on the same
      standardized 80/20 split, side-by-side with each approach's originally published performance.
      Also includes a dataset-level exploratory view: missingness, feature correlation, and PCA.
    - **Analyze Your Own Dataset** — upload your own untargeted omics dataset (any disease,
      any feature naming convention) and run the thesis's combined methodology on it end to end.

    **Before running this tool:** start the backend API in a separate terminal window:
    uvicorn api:app --reload
    Leave that window running, then launch this app with `streamlit run app.py`.
    If Tab 1 shows a connection error, check that the API terminal is still running.
    """)

tab1, tab2 = st.tabs(["Dataset Comparison", "Analyze Your Own Dataset"])

with tab1:
    st.header("Compare approaches on a standardized 80/20 split")
    st.caption(
        "Each approach was trained on 80% of a dataset and tested on that same "
        "dataset's own held-out 20%. Published AUC (where available) is shown "
        "only on the dataset each approach's own paper was actually evaluated on."
    )

    dataset_name = st.selectbox("Choose a dataset:", ["Franzosa", "Jacobs", "MetAML"], key="t1_dataset")
    api_dataset_key = dataset_name.lower()

    st.subheader(f"Results — {dataset_name}")
    filtered = MAIN_TABLE_DF[MAIN_TABLE_DF["Dataset"] == dataset_name].drop(columns=["Dataset"])
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Confusion matrix and ROC curve by approach")

    try:
        response = requests.get(f"{API_URL}/compare", params={"dataset": api_dataset_key}, timeout=10)
        response.raise_for_status()
        comparison_data = response.json()["comparison"]
    except requests.exceptions.RequestException as e:
        st.error(
            "Could not reach the MetaboIBD API. Make sure it's running in a separate "
            f"terminal with `uvicorn api:app --reload`.\n\nDetails: {e}"
        )
        comparison_data = None

    if comparison_data:
        selected_approach = st.selectbox(
            "Choose an approach to view its confusion matrix and ROC curve:",
            [row["methodology"] for row in comparison_data],
            format_func=lambda x: x.capitalize(),
            key="t1_cm_approach",
        )
        st.caption(approach_info.get(selected_approach, ""))

        cm_row = next(r for r in comparison_data if r["methodology"] == selected_approach)
        cm = np.array(cm_row["confusion_matrix"])
        class_names = [str(i) for i in range(cm.shape[0])]

        col_cm, col_roc = st.columns(2)

        with col_cm:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(cm, cmap="Greens")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(f"Confusion Matrix — {selected_approach.capitalize()}", fontsize=9)
            st.pyplot(fig, use_container_width=False)

        with col_roc:
            y_test_roc = np.array(cm_row["y_test"])
            probs_roc = np.array(cm_row["probabilities"])
            n_classes_roc = probs_roc.shape[1]

            fig2, ax2 = plt.subplots(figsize=(3, 3))
            if n_classes_roc == 2:
                fpr, tpr, _ = roc_curve(y_test_roc, probs_roc[:, 1])
                auc_val = roc_auc_score(y_test_roc, probs_roc[:, 1])
                ax2.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
            else:
                for class_idx in range(n_classes_roc):
                    y_bin = (y_test_roc == class_idx).astype(int)
                    fpr, tpr, _ = roc_curve(y_bin, probs_roc[:, class_idx])
                    auc_val = roc_auc_score(y_bin, probs_roc[:, class_idx])
                    ax2.plot(fpr, tpr, label=f"Class {class_idx} (AUC={auc_val:.3f})", linewidth=1.5)
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax2.set_xlabel("False Positive Rate", fontsize=8)
            ax2.set_ylabel("True Positive Rate", fontsize=8)
            ax2.set_title(f"ROC Curve — {selected_approach.capitalize()}", fontsize=9)
            ax2.legend(fontsize=7)
            ax2.tick_params(labelsize=7)
            st.pyplot(fig2, use_container_width=False)

        st.subheader("Ranked / most important features")
        st.caption(
            "Feature importance from the trained Random Forest — how much each feature "
            "contributed to the model's classification decisions, relative to all other "
            "features (importances sum to 1.0 across the full feature set)."
        )
        try:
            rf_response = requests.get(
                f"{API_URL}/analysis/ranked-features",
                params={"dataset": api_dataset_key, "approach": selected_approach},
                timeout=10,
            )
            rf_response.raise_for_status()
            ranked_features = rf_response.json()["ranked_features"]
            ranked_df = pd.DataFrame(ranked_features).head(15).sort_values("importance")

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.barh(ranked_df["feature"], ranked_df["importance"], color="#2e7d32")
            ax3.set_xlabel("Importance", fontsize=9)
            ax3.set_title(f"Top 15 Features — {selected_approach.capitalize()}", fontsize=10)
            ax3.tick_params(labelsize=7)
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=False)
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not load ranked features: {e}")

    st.divider()
    st.subheader("Dataset-level exploratory analysis")
    st.caption(f"These views describe the full {dataset_name} dataset, before any approach-specific filtering.")

    col_miss, col_pca = st.columns(2)

    with col_miss:
        st.markdown("**Missingness**")
        try:
            miss_response = requests.get(f"{API_URL}/analysis/missingness", params={"dataset": api_dataset_key}, timeout=10)
            miss_response.raise_for_status()
            missingness = miss_response.json()["missingness"]
            if missingness:
                st.dataframe(pd.DataFrame(missingness), use_container_width=True, hide_index=True)
            else:
                st.info("No missing values were found in this dataset.")
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not load missingness report: {e}")

        st.markdown("**Top correlated feature pairs**")
        try:
            corr_response = requests.get(f"{API_URL}/analysis/correlation", params={"dataset": api_dataset_key}, timeout=10)
            corr_response.raise_for_status()
            correlations = corr_response.json()["correlations"]
            st.dataframe(pd.DataFrame(correlations).head(10), use_container_width=True, hide_index=True)
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not load correlation report: {e}")

    with col_pca:
        st.markdown("**PCA (all samples, CLR-transformed)**")
        try:
            pca_response = requests.get(f"{API_URL}/analysis/pca-plot", params={"dataset": api_dataset_key}, timeout=10)
            pca_response.raise_for_status()
            st.image(pca_response.content, use_container_width=True)
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not load PCA plot: {e}")

with tab2:
    st.header("Analyze Your Own Dataset")
    st.write(
        "Upload your own untargeted metabolomics (or other omics) dataset — any disease, "
        "any feature naming convention. This runs the full combined methodology developed "
        "in this thesis: sparsity filtering, an optional compound-identity filter, CLR "
        "transformation, outlier removal, adaptive feature selection, and a 4-classifier "
        "comparison (Random Forest, XGBoost, Ridge, LASSO) — trained and tested fresh on "
        "your data, end to end."
    )

    with st.expander("See an example of the expected file format", expanded=False):
        st.markdown("Your uploaded files should roughly follow this shape - the app auto-detects your actual column names, so exact naming isn't required. Below is a real excerpt from the Jacobs dataset used in this thesis.")
        st.markdown("**Features file** (samples as rows, one column per feature):")
        example_features = {
            "Sample": ["A001", "A002"],
            "Positive_100.0749_3.2586": [2.836639676, 0],
            "Positive_100.0751_7.3785": [0, 0],
            "Positive_100.0752_0.841": [0, 0],
            "Positive_100.0752_7.6938": [0, 0],
            "Positive_100.0755_8.6787": [2.309716599, 0],
        }
        st.dataframe(example_features, use_container_width=True, hide_index=True)
        st.markdown("**Metadata file** (one row per sample, with a disease/outcome-label column):")
        example_metadata = {
            "Dataset": ["JACOBS_IBD_FAMILIES_2016", "JACOBS_IBD_FAMILIES_2016"],
            "Sample": ["A001", "A002"],
            "Subject": ["A001", "A002"],
            "Study.Group": ["CD", "CD"],
            "Age": [18, 18],
            "Gender": ["Female", "Female"],
        }
        st.dataframe(example_metadata, use_container_width=True, hide_index=True)

    st.subheader("1. Upload your files")
    col1, col2 = st.columns(2)
    with col1:
        features_upload = st.file_uploader(
            "Feature/metabolite file (samples x features, .tsv or .csv)",
            type=["tsv", "csv"], key="universal_features"
        )
    with col2:
        metadata_upload = st.file_uploader(
            "Metadata file (sample IDs + disease label, .tsv or .csv)",
            type=["tsv", "csv"], key="universal_metadata"
        )

    mapping_upload = st.file_uploader(
        "Compound-mapping file (optional — only if your data has HMDB/KEGG/compound "
        "identity annotations)",
        type=["tsv", "csv"], key="universal_mapping"
    )

    st.subheader("2. Run the analysis")
    st.caption("This can take a few minutes — grid search runs over both Random Forest and XGBoost.")

    if st.button("Run Analysis", key="universal_run", disabled=(features_upload is None or metadata_upload is None)):
        if features_upload is None or metadata_upload is None:
            st.error("Please upload both a features file and a metadata file before running the analysis.")
            st.stop()
        files = {
            "features_file": (features_upload.name, features_upload.getvalue()),
            "metadata_file": (metadata_upload.name, metadata_upload.getvalue()),
        }
        if mapping_upload is not None:
            files["mapping_file"] = (mapping_upload.name, mapping_upload.getvalue())

        with st.spinner("Running the full pipeline on your dataset..."):
            try:
                response = requests.post(f"{API_URL}/universal/analyze", files=files, timeout=1900)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Analysis failed: {e}")
                result = None

        if result:
            st.success(f"Done! Best model: **{result['best_model']}** (AUC = {result['best_auc']})")

            if result.get("generated_mapping"):
                gm = result["generated_mapping"]
                st.subheader("Compound mapping")
                st.write(f"{gm['matched_features']} of {gm['total_features']} features were matched to a known compound.")
                if gm["preview"]:
                    preview_df = pd.DataFrame(gm["preview"])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download full generated mapping (TSV)",
                        data=preview_df.to_csv(sep="\t", index=False),
                        file_name="generated_mapping.tsv",
                        mime="text/tab-separated-values",
                        key="download_mapping",
                    )
                else:
                    st.info("No features could be matched to a known compound for this dataset.")
            elif mapping_upload is None:
                st.info("No mapping file was provided and no compound identities could be auto-generated for this dataset's feature names.")

            st.subheader("Results by classifier")
            results_df = pd.DataFrame(result["results"])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            with st.expander("Pipeline log (auto-detected columns, feature counts, branch taken)"):
                st.text(result["log"])