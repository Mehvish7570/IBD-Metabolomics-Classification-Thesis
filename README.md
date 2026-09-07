# IBD-Metabolomics-Classification-Thesis
Code, data, and supplementary materials for MSc thesis: Machine Learning for Classification on Untargeted Metabolomics Data
# Machine Learning for Classification on Untargeted Metabolomics Data
### A Comparative Study Centered on the PRISM IBD Dataset

This repository contains the code, data, figures, and supplementary 
material accompanying the MSc thesis of the same title, submitted to the 
University of Birmingham.

## Repository Structure

- `data/` — the three discovery datasets used in this thesis (Franzosa, 
  Jacobs, MetAML), provided as ZIP archives for data availability.
- `code/` — the core analysis scripts:
  - `own_method_final_franzosa.py` — the final combined methodology, using 
    the dataset's existing compound-mapping file. Produces the official 
    reported result (AUC = 0.9076).
  - `best_exploratory_approach_franzosa.ipynb` / `_jacobs.ipynb` / 
    `_metaml.ipynb` — the best-performing result from the earlier 
    exploratory phase of method development, for each dataset.
  - `own_method_universal_restructured_supervisor_v2.py` — the final 
    combined methodology restructured into a dataset-agnostic pipeline 
    that generates its own compound-identity mapping directly from feature 
    mass, with no pre-existing annotation required.
  - `hmp2_external_validation_v2.py` — external validation of the combined 
    methodology on the independent HMP2 cohort.
- `figures/` — all main-text and supplementary figures generated for this 
  thesis, including performance comparisons, ROC curves, radar charts, 
  heatmaps, boxplots, the correlation network diagram, and the Mummichog 
  pathway-enrichment figure.
- `mummichog/` — full Mummichog pathway-enrichment output for the Franzosa 
  dataset.
- `key_resources/` — a table of software packages and versions used 
  throughout this study, for reproducibility.

## Data Availability

The three discovery datasets are provided in `data/` as ZIP archives.

## How to Run

Each script can be run directly with Python 3.9+. Required packages and 
versions are listed in `key_resources/key_resources_table.csv`.

## Author

Mehvish Shaikh — MSc Health Data Science, University of Birmingham
