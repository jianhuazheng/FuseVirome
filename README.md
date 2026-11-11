# FuseVirome

A machine-learning framework with dynamic reference construction for identifying viral signals in veterinary metatranscriptomic sequencing data.

## Overview

FuseVirome integrates alignment-derived features from three evidence tracks—**read coverage**, **contig coverage**, and a **fused read+contig track**—together with BLAST annotation statistics. The pipeline performs two reproducible feature-refinement steps, compares six classifiers, runs an ablation study across eleven feature configurations, and generates manuscript figures and performance metrics.

This repository contains the complete reproducible analysis code, the raw and aggregated datasets, and the manuscript figures.

## Repository Structure

```
FuseVirome/
├── src/                          # Source code
│   ├── data_preparation.py       # Raw data cleaning & aggregation
│   ├── fusevirome.py             # ML pipeline, ablation & figures
│   └── shap_experiment.py        # SHAP model interpretability experiment
├── data/
│   ├── raw/                      # Raw alignment outputs
│   │   ├── data_base.csv         # Training raw data
│   │   └── data_test.csv         # External validation raw data
│   └── processed/                # Sample-level aggregated features
│       ├── Aggregated_Training_Data.csv
│       └── Aggregated_External_Validation_Data.csv
├── figures/                      # Curated manuscript reference figures
├── outputs/figures/              # Generated results (ignored by Git)
├── docs/
│   ├── manuscript.docx           # Main manuscript
│   └── supplementary.docx        # Supplementary materials
├── tests/                        # Lightweight reproducibility tests
├── .github/workflows/            # Continuous-integration checks
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
```

## Installation

```bash
# Clone or download the repository, then install dependencies
pip install -r requirements.txt
```

Tested with Python 3.9+. The code is Python-based and does not require a JAR
or Java runtime.

## Usage

### 1. Prepare aggregated data from raw alignment outputs

```bash
python src/data_preparation.py
```

This reads `data/raw/data_base.csv` and `data/raw/data_test.csv`, renames Chinese column names to English identifiers, parses size/percent fields, and aggregates contig-level records to sample-level feature matrices saved in `data/processed/`.

### 2. Run the full ML pipeline

```bash
python src/fusevirome.py
```

Use `--output-dir PATH` to write generated figures to a different location.

The script:
- Loads the aggregated training and external-validation data.
- Applies constant/non-numeric removal and redundant fused-track exclusion.
- Trains and evaluates six classifiers (Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, k-NN, Decision Tree).
- Performs an 11-configuration ablation study.
- Computes permutation importance, feature stability, and PPV/NPV.
- Generates ROC, feature-importance, bootstrap AUC, calibration, DCA, F1 bar-chart, and metrics boxplot figures in `outputs/figures/`.

## Data Dictionary

Key feature groups in the aggregated files:

| Group | Example Features |
|-------|------------------|
| Read coverage | `Reads_Count_reads`, `Coverage_bp_reads`, `Coverage_Percent_reads`, `Avg_Coverage_Depth_x_reads`, `Mean_BaseQ_reads`, `Mean_MapQ_reads`, `Histo_Max_Bin_Percent_reads` |
| Contig coverage | `Reads_Count_contigs`, `Coverage_bp_contigs`, `Coverage_Percent_contigs`, `Avg_Coverage_Depth_x_contigs`, `Mean_BaseQ_contigs`, `Mean_MapQ_contigs`, `Histo_Max_Bin_Percent_contigs` |
| Fused track | `*_fused` counterparts of the above |
| De novo contigs | `Count_Total_Contigs`, `Count_Correct_Contigs`, `Count_Noise_Contigs` |
| BLAST quality | `identity_max/mean`, `BitScore_max/mean/sum`, `AlignLength_max/sum`, `qlen_mean`, `Evalue_min`, `Mismatches_mean`, `Align_Qlen_Ratio_Max` |

Target columns:
- `Y`: binary label (1 = positive, 0 = negative)
- `Y_label_max`: original qPCR-derived label before binarization

## Figures

The `figures/` directory contains:

- `Figure 1.jpg`
- `Figure 2.png` – ROC curves of six classifiers
- `Figure 3.png` – Top 10 feature importance
- `Figure 4.png` – Original combined figure from the manuscript
- `Figure 5.png` – Decision curve analysis
- `Figure S1.png`, `Figure S2A.png`, `Figure S2B.png`, and
  `Figure S3.png` – `Figure S7.png` – supplementary figures

Running `fusevirome.py` writes the following reproducible outputs under
`outputs/figures/`:

- `Figure2_ROC.png`
- `Figure3_Features.png`
- `Figure3B_Feature_Category_Summary.png`
- `Figure4A_Bootstrap_AUC.png`
- `Figure4B_Calibration_Curve.png`
- `Figure5_DCA.png`
- `Figure6_F1_BarChart.png`

### 3. Run the SHAP interpretability experiment

```bash
python src/shap_experiment.py
```

This command also accepts `--output-dir PATH`.

This script explains the final 29-feature Random Forest model using SHAP (`shap` package). It reproduces the same train/test split as `fusevirome.py`, computes TreeExplainer SHAP values, compares Gini / permutation / SHAP importance metrics, and generates global summary, beeswarm, dependence, and local waterfall plots.

Outputs saved to `outputs/figures/`:
- `Figure8_SHAP_Global_Summary.png`
- `Figure9_SHAP_Beeswarm.png`
- `Figure10_SHAP_Importance_Comparison.png`
- `Figure11_SHAP_Dependence_TopFeatures.png`
- `Figure12_SHAP_Local_Waterfall.png` (+ `.pdf` and `.tif` variants)
- `shap_importance.csv`
- `shap_values.npy`

## Citation

If you use FuseVirome in your research, please cite the accompanying manuscript:

> FuseVirome: A Machine Learning Framework with Dynamic Reference Construction for Viral Identification in Veterinary Metatranscriptomics.

See `docs/manuscript.docx` and `CITATION.cff` for details.

## Reproducibility checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs the same tests on every push and pull request.

## License

The source code is released under the MIT License. See [LICENSE](LICENSE) for
details. Before publishing the repository, confirm that the datasets, figures,
manuscript, and supplementary material may be redistributed under the terms of
their original study and data providers.

## Contact

For questions about the code or data, please open an issue in the GitHub repository.
