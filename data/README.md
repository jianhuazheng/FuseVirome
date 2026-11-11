# Data files

## Raw inputs

| File | Purpose | Encoding |
|---|---|---|
| `raw/data_base.csv` | Training alignment records | GBK |
| `raw/data_test.csv` | External-validation alignment records | GBK |

## Processed inputs

| File | Rows | Purpose | Encoding |
|---|---:|---|---|
| `processed/Aggregated_Training_Data.csv` | 120 | Model development | UTF-8 |
| `processed/Aggregated_External_Validation_Data.csv` | 179 | Independent validation | UTF-8 |

The processed tables contain 40 columns, including identifiers, targets, and
aggregated model features. `src/data_preparation.py` rebuilds these files from
the raw tables.

`Y` is the binary model target. `Y_label_max` and any qPCR-derived labels are
retained for provenance but removed before model fitting to prevent target
leakage.

Before publishing this repository, confirm that sample identifiers and all
source datasets may be redistributed under the applicable study approvals and
data-provider terms.
