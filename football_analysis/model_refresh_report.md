# Football Model Refresh Report

## What Changed

- Replaced the notebook's ad hoc preprocessing with one canonical pipeline.
- Normalized `Nation`, `Comp`, and primary `Pos`.
- Built a season-aware `combined_base` dataset by appending `top5-players.csv` and `players_data-2024_2025.csv` instead of trying to merge rows directly.
- Added derived per-90 features and `season` / `source_file` columns.
- Built a separate `detailed_2024_2025` experiment after dropping 76 duplicated metadata columns such as `Pos_stats_*`, `Nation_stats_*`, `Comp_stats_*`, and similar leakage sources.
- Replaced manual dataframe label encoding with sklearn pipelines using median imputation, one-hot encoding, and model-specific scaling.
- Replaced the notebook's rounded regression scoring with proper continuous-value evaluation.

## Baseline Vs Improved Results

### Original notebook baselines

- Random Forest classification holdout accuracy: `0.7567`
- Random Forest classification 5-fold CV mean accuracy: `0.7394`
- Linear Regression regression holdout MSE without rounding: `1.3892`
- Linear Regression regression holdout R² without rounding: `0.8625`

### New classification results

- `combined_base` best model: HistGradientBoosting
  - Holdout accuracy: `0.8063`
  - Holdout macro-F1: `0.8422`
  - Holdout MF recall: `0.6877`
  - CV mean accuracy: `0.8093`
- `detailed_2024_2025` best model: HistGradientBoosting
  - Holdout accuracy: `0.9093`
  - Holdout macro-F1: `0.9175`
  - Holdout MF recall: `0.8026`
  - CV mean accuracy: `0.9003`

### New regression results

- Best model: ElasticNet
  - Holdout MAE: `0.7805`
  - Holdout MSE: `1.3664`
  - Holdout R²: `0.8761`

## Why Accuracy Is Still Not Perfect

### 1. The target labels are noisy

After filtering to `Min > 90`, the raw data still contains many mixed-role labels:

- `top5-players.csv`: `721 / 2426` rows are mixed-position labels (`29.7%`)
- `players_data-2024_2025.csv`: `734 / 2476` rows are mixed-position labels (`29.6%`)

Examples include `FW,MF`, `MF,FW`, `DF,MF`, and `MF,DF`. The workflow collapses those to the first position so the model is trained on labels that already hide the player's hybrid role.

### 2. Midfield is the hardest class

For the best `combined_base` model:

- MF recall is only `0.6877`
- GK recall is effectively perfect on the holdout set

That means the hardest problem is not identifying goalkeepers. It is separating midfielders from defenders and forwards. In the combined dataset, the best model still misclassifies many midfielders as defenders or forwards.

### 3. The safe merged dataset is intentionally conservative

The cross-season `combined_base` table uses only aligned shared columns plus derived per-90 features. That is the correct leakage-safe way to append seasons, but it also drops a lot of rich signal that exists only in the detailed 2024/2025 file.

This shows up directly in the scores:

- `combined_base` holdout accuracy: `0.8063`
- `detailed_2024_2025` holdout accuracy: `0.9093`

So a large part of the remaining error comes from missing feature detail, not just model weakness.

### 4. Position is tactical, not purely statistical

The CSV files contain aggregate stat summaries. They do not fully encode:

- formation context
- side of the pitch
- possession role
- coach instructions
- out-of-possession shape
- match-state context

Two players can post similar attacking and progression numbers while still being labeled differently because their tactical role differs.

### 5. Adjacent football roles naturally overlap

Defenders, midfielders, and forwards do not form clean statistical clusters, especially for:

- full-backs vs wide midfielders
- defensive midfielders vs center-backs stepping into build-up
- attacking midfielders vs second strikers / wide forwards

That overlap limits achievable accuracy even after leakage is removed and model quality improves.

### 6. Same-season statistics still contain role/outcome entanglement

This project is intentionally same-season benchmarking, not future forecasting. That means we are predicting a player's same-season position or goals from same-season stat lines. That setup is valid for benchmarking and exploratory modeling, but it also means the model is learning from outcomes that reflect the role rather than fully explaining it.

## Bottom Line

The original notebook was mainly limited by feature design and evaluation quality. The refresh fixes the leakage risk, aligns the data correctly across seasons, and improves both classification and regression. The remaining classification error is mostly due to noisy hybrid labels, limited shared features in the combined dataset, and the fact that tactical position cannot be perfectly recovered from aggregate stat tables alone.
