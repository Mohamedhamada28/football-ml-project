from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    TweedieRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "model_refresh"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"

TOP5_PATH = ROOT / "top5-players.csv"
DETAIL_2024_2025_PATH = ROOT / "players_data-2024_2025.csv"

NOTEBOOK_CLASSIFICATION_FEATURES = [
    "Gls",
    "Ast",
    "PK",
    "CrdY",
    "CrdR",
    "xG",
    "PrgC",
    "PrgP",
    "PrgR",
]

SAFE_SHARED_NUMERIC = [
    "Age",
    "Born",
    "MP",
    "Starts",
    "Min",
    "90s",
    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "xG",
    "npxG",
    "xAG",
    "npxG+xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]

PER90_NUMERIC = [
    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "xG",
    "npxG",
    "xAG",
    "npxG+xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]

DETAILED_METADATA_DUPLICATE_BASES = {
    "Rk",
    "Nation",
    "Pos",
    "Comp",
    "Age",
    "Born",
    "MP",
    "Starts",
    "Min",
    "90s",
}

REGRESSION_TARGET_LEAKAGE_COLUMNS = {
    "Gls",
    "Gls_per90",
    "G+A",
    "G+A_per90",
    "G-PK",
    "G-PK_per90",
    "G+A_90",
    "G-PK_90",
    "G+A-PK_90",
}

CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "mf_recall": make_scorer(
        recall_score,
        labels=["MF"],
        average="macro",
        zero_division=0,
    ),
}

REGRESSION_SCORING = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export_model_artifact(model: Pipeline, filename: str) -> Path:
    output_path = MODEL_OUTPUT_DIR / filename
    dump(model, output_path)
    return output_path


def normalize_nation(series: pd.Series) -> pd.Series:
    parts = series.fillna("").astype(str).str.split(" ", n=1, expand=True)
    if parts.shape[1] == 1:
        return parts[0].replace("", pd.NA)
    return parts[1].replace("", pd.NA)


def normalize_comp(series: pd.Series) -> pd.Series:
    parts = series.fillna("").astype(str).str.split(" ", n=1, expand=True)
    if parts.shape[1] == 1:
        return parts[0].replace("", pd.NA)
    return parts[1].replace("", pd.NA)


def normalize_primary_position(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split(",", n=1, expand=True)[0].str.strip().replace("", pd.NA)


def canonicalize_common_fields(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Nation"] = normalize_nation(df["Nation"])
    df["Comp"] = normalize_comp(df["Comp"])
    df["Pos"] = normalize_primary_position(df["Pos"])
    return df


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator.astype(float) / denom.astype(float)


def add_per90_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    enriched = df.copy()
    for column in columns:
        enriched[f"{column}_per90"] = safe_rate(enriched[column], enriched["90s"])
    enriched["start_rate"] = safe_rate(enriched["Starts"], enriched["MP"])
    return enriched


def load_canonical_season(path: Path, season: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = canonicalize_common_fields(raw)
    raw = raw[raw["Min"] > 90].copy()
    raw["season"] = season
    raw["source_file"] = path.name
    raw = add_per90_features(raw, PER90_NUMERIC)
    keep_columns = (
        ["Player", "Nation", "Squad", "Comp", "Pos", "season", "source_file"]
        + SAFE_SHARED_NUMERIC
        + [f"{column}_per90" for column in PER90_NUMERIC]
        + ["start_rate"]
    )
    return raw[keep_columns].copy()


def build_combined_base() -> pd.DataFrame:
    current = load_canonical_season(TOP5_PATH, "2023_2024")
    detailed = load_canonical_season(DETAIL_2024_2025_PATH, "2024_2025")
    return pd.concat([current, detailed], ignore_index=True)


def find_detailed_duplicate_metadata_columns(columns: list[str]) -> list[str]:
    duplicates: list[str] = []
    for column in columns:
        if "_" not in column:
            continue
        base = column.split("_", 1)[0]
        if base in DETAILED_METADATA_DUPLICATE_BASES:
            duplicates.append(column)
    return sorted(set(duplicates))


def build_detailed_2024_2025() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DETAIL_2024_2025_PATH)
    df = canonicalize_common_fields(df)
    df = df[df["Min"] > 90].copy()
    df["season"] = "2024_2025"
    df["source_file"] = DETAIL_2024_2025_PATH.name
    duplicated_metadata = find_detailed_duplicate_metadata_columns(df.columns.tolist())
    df = df.drop(columns=duplicated_metadata, errors="ignore")
    return df, duplicated_metadata


def make_preprocessors(X: pd.DataFrame) -> dict[str, ColumnTransformer]:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    dense_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    return {
        "linear": linear_preprocessor,
        "tree": tree_preprocessor,
        "dense": dense_preprocessor,
    }


def make_classification_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessors = make_preprocessors(X)
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessors["linear"]),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessors["tree"]),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=42,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("preprocessor", preprocessors["tree"]),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=150,
                        random_state=42,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("preprocessor", preprocessors["dense"]),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=150,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def make_regression_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessors = make_preprocessors(X)
    return {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessors["linear"]),
                ("model", LinearRegression()),
            ]
        ),
        "ElasticNet": Pipeline(
            steps=[
                ("preprocessor", preprocessors["linear"]),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessors["tree"]),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessors["dense"]),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=150,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "PoissonRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessors["linear"]),
                ("model", PoissonRegressor(alpha=0.1, max_iter=300)),
            ]
        ),
        "TweedieRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessors["linear"]),
                ("model", TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=300)),
            ]
        ),
    }


def flatten_classification_report(
    report: dict[str, Any],
    experiment: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "experiment": experiment,
                "model": model_name,
                "label": label,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1-score"),
                "support": metrics.get("support"),
            }
        )
    return rows


def flatten_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    experiment: str,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return rows


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def extract_feature_rows(
    pipeline: Pipeline,
    experiment: str,
    model_name: str,
    task: str,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(preprocessor, "get_feature_names_out"):
        return []

    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    rows: list[dict[str, Any]] = []

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            ranking = np.argsort(np.abs(coef))[::-1][:top_n]
            for rank, index in enumerate(ranking, start=1):
                rows.append(
                    {
                        "task": task,
                        "experiment": experiment,
                        "model": model_name,
                        "target_class": "__regression__",
                        "feature": feature_names[index],
                        "view_type": "coefficient",
                        "value": float(coef[index]),
                        "rank": rank,
                    }
                )
        else:
            classes = getattr(model, "classes_", [f"class_{i}" for i in range(coef.shape[0])])
            for class_index, class_name in enumerate(classes):
                class_coef = coef[class_index]
                ranking = np.argsort(np.abs(class_coef))[::-1][:top_n]
                for rank, index in enumerate(ranking, start=1):
                    rows.append(
                        {
                            "task": task,
                            "experiment": experiment,
                            "model": model_name,
                            "target_class": class_name,
                            "feature": feature_names[index],
                            "view_type": "coefficient",
                            "value": float(class_coef[index]),
                            "rank": rank,
                        }
                    )

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        ranking = np.argsort(importance)[::-1][:top_n]
        for rank, index in enumerate(ranking, start=1):
            rows.append(
                {
                    "task": task,
                    "experiment": experiment,
                    "model": model_name,
                    "target_class": "__global__",
                    "feature": feature_names[index],
                    "view_type": "importance",
                    "value": float(importance[index]),
                    "rank": rank,
                }
            )

    return rows


def render_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def assert_no_detailed_leakage(columns: list[str]) -> None:
    bad_columns = []
    for column in columns:
        if column == "Pos" or column.startswith("Pos_"):
            bad_columns.append(column)
            continue
        if "_" in column and column.split("_", 1)[0] in DETAILED_METADATA_DUPLICATE_BASES:
            bad_columns.append(column)
    if bad_columns:
        raise ValueError(f"Leakage columns still present: {sorted(set(bad_columns))}")


def guard_against_suspicious_accuracy(summary: pd.DataFrame, threshold: float = 0.98) -> None:
    suspicious = summary[
        (summary["holdout_accuracy"] >= threshold) | (summary["cv_accuracy_mean"] >= threshold)
    ]
    if not suspicious.empty:
        offenders = suspicious[["experiment", "model", "holdout_accuracy", "cv_accuracy_mean"]]
        raise RuntimeError(
            "Suspiciously high classification accuracy detected. Review for leakage first.\n"
            f"{offenders.to_string(index=False)}"
        )


def evaluate_classification_experiment(
    experiment: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    models = make_classification_models(X)
    labels = sorted(y.dropna().unique().tolist())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    summary_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    confusion_lookup: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        print(f"[classification] {experiment}: {model_name}")
        cv_scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=CLASSIFICATION_SCORING,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        summary_rows.append(
            {
                "task": "classification",
                "experiment": experiment,
                "model": model_name,
                "holdout_accuracy": accuracy_score(y_test, predictions),
                "holdout_macro_f1": f1_score(y_test, predictions, average="macro"),
                "holdout_mf_recall": recall_score(
                    y_test,
                    predictions,
                    labels=["MF"],
                    average="macro",
                    zero_division=0,
                ),
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
                "cv_macro_f1_mean": float(np.mean(cv_scores["test_macro_f1"])),
                "cv_macro_f1_std": float(np.std(cv_scores["test_macro_f1"])),
                "cv_mf_recall_mean": float(np.mean(cv_scores["test_mf_recall"])),
                "cv_mf_recall_std": float(np.std(cv_scores["test_mf_recall"])),
            }
        )

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_rows.extend(flatten_classification_report(report, experiment, model_name))

        matrix = confusion_matrix(y_test, predictions, labels=labels)
        confusion_lookup[model_name] = matrix
        confusion_rows.extend(flatten_confusion_matrix(matrix, labels, experiment, model_name))

        feature_rows.extend(extract_feature_rows(model, experiment, model_name, "classification"))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_macro_f1", "holdout_accuracy"],
        ascending=[False, False],
    )
    guard_against_suspicious_accuracy(summary)

    best_row = summary.iloc[0]
    best_model_name = str(best_row["model"])
    best_matrix = confusion_lookup[best_model_name]

    best_model = make_classification_models(X)[best_model_name]
    best_model.fit(X, y)
    model_path = export_model_artifact(
        best_model,
        f"{experiment}_{best_model_name}_classification.joblib",
    )

    plot_path = OUTPUT_DIR / f"{experiment}_best_confusion_matrix.png"
    render_confusion_matrix_plot(best_matrix, labels, f"{experiment} - {best_model_name}", plot_path)

    metadata = {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "feature_columns": X.columns.tolist(),
        "target_column": y.name,
        "labels": labels,
        "plot_path": str(plot_path),
    }
    return (
        summary,
        pd.DataFrame(report_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(feature_rows),
        metadata,
    )


def evaluate_regression_experiment(
    experiment: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    models = make_regression_models(X)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        print(f"[regression] {experiment}: {model_name}")
        cv_scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=REGRESSION_SCORING,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        holdout_mse = mean_squared_error(y_test, predictions)

        summary_rows.append(
            {
                "task": "regression",
                "experiment": experiment,
                "model": model_name,
                "holdout_mae": mean_absolute_error(y_test, predictions),
                "holdout_mse": holdout_mse,
                "holdout_rmse": holdout_mse**0.5,
                "holdout_r2": r2_score(y_test, predictions),
                "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
                "cv_mae_std": float(np.std(-cv_scores["test_mae"])),
                "cv_mse_mean": float(-np.mean(cv_scores["test_mse"])),
                "cv_mse_std": float(np.std(-cv_scores["test_mse"])),
                "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
                "cv_r2_std": float(np.std(cv_scores["test_r2"])),
            }
        )
        feature_rows.extend(extract_feature_rows(model, experiment, model_name, "regression"))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_r2", "holdout_mae"],
        ascending=[False, True],
    )
    best_model_name = str(summary.iloc[0]["model"])
    best_model = make_regression_models(X)[best_model_name]
    best_model.fit(X, y)
    model_path = export_model_artifact(
        best_model,
        f"{experiment}_{best_model_name}_regression.joblib",
    )

    metadata = {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "feature_columns": X.columns.tolist(),
        "target_column": y.name,
    }
    return summary, pd.DataFrame(feature_rows), metadata


def notebook_style_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.copy()
    for source_column, encoded_column in [
        ("Nation", "Nation_num"),
        ("Pos", "Pos_num"),
        ("Squad", "Squad_num"),
        ("Player", "Player_name_num"),
        ("Comp", "Comp_num"),
    ]:
        encoder = LabelEncoder()
        baseline[encoded_column] = encoder.fit_transform(baseline[source_column].astype(str))
    return baseline


def reproduce_notebook_baselines() -> pd.DataFrame:
    raw = pd.read_csv(TOP5_PATH)
    df = canonicalize_common_fields(raw)
    df = df[df["Min"] > 90].copy()
    df = df.dropna(how="any").copy()

    baseline_rows: list[dict[str, Any]] = []

    X_classification = df[NOTEBOOK_CLASSIFICATION_FEATURES]
    y_classification = df["Pos"]
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_classification,
        y_classification,
        test_size=0.2,
        random_state=42,
    )

    notebook_rf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    notebook_rf.fit(X_train_class, y_train_class)
    rf_predictions = notebook_rf.predict(X_test_class)
    baseline_rows.append(
        {
            "task": "baseline",
            "metric": "notebook_rf_holdout_accuracy",
            "value": accuracy_score(y_test_class, rf_predictions),
        }
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        notebook_rf,
        X_classification,
        y_classification,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    baseline_rows.append(
        {
            "task": "baseline",
            "metric": "notebook_rf_cv_accuracy_mean",
            "value": float(np.mean(cv_scores["test_score"])),
        }
    )

    baseline_regression = notebook_style_regression_frame(df)
    X_regression = baseline_regression.drop(
        columns=[
            "Player",
            "Nation",
            "Comp",
            "Pos",
            "Squad",
            "Gls",
            "G+A",
            "G-PK",
            "xG",
            "npxG+xAG",
            "G+A_90",
            "G-PK_90",
            "G+A-PK_90",
            "xG_90",
            "Gls_90",
            "xG+xAG_90",
            "npxG_90",
            "npxG+xAG_90",
        ]
    )
    y_regression = baseline_regression["Gls"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression,
        y_regression,
        test_size=0.2,
        random_state=42,
    )

    notebook_linear = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    notebook_linear.fit(X_train_reg, y_train_reg)
    linear_predictions = notebook_linear.predict(X_test_reg)
    rounded_predictions = np.maximum(linear_predictions, 0).astype(int)

    baseline_rows.extend(
        [
            {
                "task": "baseline",
                "metric": "notebook_linear_holdout_mse_unrounded",
                "value": mean_squared_error(y_test_reg, linear_predictions),
            },
            {
                "task": "baseline",
                "metric": "notebook_linear_holdout_r2_unrounded",
                "value": r2_score(y_test_reg, linear_predictions),
            },
            {
                "task": "baseline",
                "metric": "notebook_linear_holdout_mse_rounded",
                "value": mean_squared_error(y_test_reg, rounded_predictions),
            },
            {
                "task": "baseline",
                "metric": "notebook_linear_holdout_r2_rounded",
                "value": r2_score(y_test_reg, rounded_predictions),
            },
        ]
    )

    return pd.DataFrame(baseline_rows)


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    ensure_output_dir()
    print("Preparing datasets...")

    combined_base = build_combined_base()
    detailed_2024_2025, dropped_metadata_columns = build_detailed_2024_2025()
    assert_no_detailed_leakage(detailed_2024_2025.drop(columns=["Pos"]).columns.tolist())

    save_dataframe(combined_base, "combined_base_training_data.csv")
    save_dataframe(detailed_2024_2025, "detailed_2024_2025_training_data.csv")

    baseline_metrics = reproduce_notebook_baselines()
    save_dataframe(baseline_metrics, "baseline_metrics.csv")
    print("Saved baselines.")

    combined_classification_X = combined_base.drop(columns=["Pos", "Player"])
    combined_classification_y = combined_base["Pos"]
    combined_summary, combined_report, combined_confusion, combined_features, combined_meta = evaluate_classification_experiment(
        experiment="combined_base",
        X=combined_classification_X,
        y=combined_classification_y,
    )

    detailed_classification_X = detailed_2024_2025.drop(columns=["Pos", "Player"])
    detailed_classification_y = detailed_2024_2025["Pos"]
    detailed_summary, detailed_report, detailed_confusion, detailed_features, detailed_meta = evaluate_classification_experiment(
        experiment="detailed_2024_2025",
        X=detailed_classification_X,
        y=detailed_classification_y,
    )

    classification_summary = pd.concat([combined_summary, detailed_summary], ignore_index=True)
    classification_reports = pd.concat([combined_report, detailed_report], ignore_index=True)
    classification_confusions = pd.concat([combined_confusion, detailed_confusion], ignore_index=True)
    classification_features = pd.concat([combined_features, detailed_features], ignore_index=True)

    save_dataframe(classification_summary, "classification_summary.csv")
    save_dataframe(classification_reports, "classification_per_class_metrics.csv")
    save_dataframe(classification_confusions, "classification_confusion_matrices.csv")
    save_dataframe(classification_features, "classification_feature_views.csv")
    print("Saved classification outputs.")

    regression_frame = combined_base.copy()
    regression_X = regression_frame.drop(
        columns=["Player"] + sorted(REGRESSION_TARGET_LEAKAGE_COLUMNS),
        errors="ignore",
    )
    regression_y = regression_frame["Gls"]
    regression_summary, regression_features, regression_meta = evaluate_regression_experiment(
        experiment="combined_base_regression",
        X=regression_X,
        y=regression_y,
    )

    save_dataframe(regression_summary, "regression_summary.csv")
    save_dataframe(regression_features, "regression_feature_views.csv")

    run_metadata = {
        "combined_base_shape": list(combined_base.shape),
        "detailed_2024_2025_shape": list(detailed_2024_2025.shape),
        "detailed_dropped_metadata_columns": dropped_metadata_columns,
        "best_models": {
            "combined_base_classification": combined_meta["best_model"],
            "detailed_2024_2025_classification": detailed_meta["best_model"],
            "combined_base_regression": regression_meta["best_model"],
        },
        "confusion_matrix_plots": {
            "combined_base": combined_meta["plot_path"],
            "detailed_2024_2025": detailed_meta["plot_path"],
        },
        "exported_best_models": {
            "combined_base_classification": combined_meta["model_path"],
            "detailed_2024_2025_classification": detailed_meta["model_path"],
            "combined_base_regression": regression_meta["model_path"],
        },
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("Saved outputs to:", OUTPUT_DIR)
    print("Best combined_base classification model:", combined_meta["best_model"])
    print("Best detailed_2024_2025 classification model:", detailed_meta["best_model"])
    print("Best combined_base regression model:", regression_meta["best_model"])
    print("Exported combined_base classification model:", combined_meta["model_path"])
    print("Exported detailed_2024_2025 classification model:", detailed_meta["model_path"])
    print("Exported combined_base regression model:", regression_meta["model_path"])


if __name__ == "__main__":
    main()
