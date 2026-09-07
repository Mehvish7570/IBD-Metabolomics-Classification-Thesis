# IBD-Metabolomics-Classification-Thesis
Code, data, and supplementary materials for MSc thesis: Machine Learning for Classification on Untargeted Metabolomics Data
# Machine Learning for Classification on Untargeted Metabolomics Data
### A Comparative Study Centered on the PRISM IBD Dataset

This repository contains the code, generated mapping files, and supplementary 
material accompanying the MSc thesis of the same title, submitted to the 
University of Birmingham.

## Repository Structure

- `data/` — the three discovery datasets used in this thesis (Franzosa, 
  Jacobs, MetAML), provided as ZIP archives for data availability.
- `code/` — the core analysis scripts:
  - `best_exploratory_method.ipynb` — the best-performing result from the 
    earlier exploratory phase of method development.
  - `combined_method.py` — the final combined methodology, using each 
    dataset's own existing compound-mapping file.
  - `combined_method_own_mapping.py` — the final combined methodology using 
    the pipeline's own self-generated compound-identity mapping (mass-based, 
    no pre-existing annotation required).
- `mapping/` — the code and output for auto-generating a compound-identity 
  mapping directly from feature mass, applied to the Franzosa dataset.
- `mummichog/` — Mummichog pathway-enrichment results for the Franzosa 
  dataset.
- `supplementary_figures/` — figures referenced in the thesis's 
  Supplementary Material but not included in the main report due to the 
  10-display-item limit: radar charts (classifier performance comparison), 
  the correlation network diagram (Franzosa), and the key resources table.

## Data Availability

The three discovery datasets are provided in `data/` as ZIP archives. 
Original sources:
- Franzosa et al. (PRISM cohort)
- Jacobs et al. (paediatric family cohort)
- Pasolli et al. / MetAML (metagenomic benchmark dataset)

## How to Run

Each script in `code/` and `mapping/` can be run directly with Python 3.9+. 
Required packages: pandas, numpy, scikit-learn, xgboost, boruta, 
imbalanced-learn, scipy, statsmodels.

## Author

Mehvish Shaikh — MSc Health Data Science, University of Birmingham
