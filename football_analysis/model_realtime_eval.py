from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from football_model_refresh import (
    REGRESSION_TARGET_LEAKAGE_COLUMNS,
    build_combined_base,
    build_detailed_2024_2025,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "model_refresh" / "models"

COMBINED_MODEL_PATH = MODEL_DIR / "combined_base_HistGradientBoosting_classification.joblib"
DETAILED_MODEL_PATH = MODEL_DIR / "detailed_2024_2025_HistGradientBoosting_classification.joblib"
REGRESSION_MODEL_PATH = MODEL_DIR / "combined_base_regression_ElasticNet_regression.joblib"


def load_models() -> tuple[object, object, object]:
    for path in [COMBINED_MODEL_PATH, DETAILED_MODEL_PATH, REGRESSION_MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
    return (
        joblib.load(COMBINED_MODEL_PATH),
        joblib.load(DETAILED_MODEL_PATH),
        joblib.load(REGRESSION_MODEL_PATH),
    )


def build_frames() -> tuple[object, object]:
    combined_base = build_combined_base()
    detailed_2024_2025, _ = build_detailed_2024_2025()
    return combined_base, detailed_2024_2025


def sample_predictions(dataset: str, n: int, seed: int) -> dict:
    combined_clf, detailed_clf, _ = load_models()
    combined_base, detailed_2024_2025 = build_frames()

    if dataset == "combined":
        frame = combined_base
        model = combined_clf
    else:
        frame = detailed_2024_2025
        model = detailed_clf

    X = frame.drop(columns=["Pos", "Player"])
    y = frame["Pos"]

    sample_X = X.sample(n=min(n, len(X)), random_state=seed)
    true_y = y.loc[sample_X.index]
    pred_y = model.predict(sample_X)

    rows = []
    for idx, true_label, pred_label in zip(sample_X.index.tolist(), true_y.tolist(), pred_y.tolist()):
        rows.append({"index": int(idx), "true_pos": str(true_label), "pred_pos": str(pred_label)})

    return {
        "mode": "sample",
        "dataset": dataset,
        "sample_size": len(rows),
        "rows": rows,
    }


def evaluate_classification(dataset: str, test_size: float, seed: int) -> dict:
    combined_clf, detailed_clf, _ = load_models()
    combined_base, detailed_2024_2025 = build_frames()

    if dataset == "combined":
        frame = combined_base
        model = combined_clf
        model_name = "combined_base_HistGradientBoosting"
    else:
        frame = detailed_2024_2025
        model = detailed_clf
        model_name = "detailed_2024_2025_HistGradientBoosting"

    X = frame.drop(columns=["Pos", "Player"])
    y = frame["Pos"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    train_pred = fitted_model.predict(X_train)
    y_pred = fitted_model.predict(X_test)
    labels = sorted(y.unique().tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    cv_scores = cross_val_score(clone(model), X, y, cv=5, scoring="accuracy")

    per_class_accuracy = {}
    for index, label in enumerate(labels):
        row_total = int(matrix[index].sum())
        per_class_accuracy[label] = float(matrix[index, index] / row_total) if row_total else 0.0

    return {
        "mode": "classification",
        "dataset": dataset,
        "model": model_name,
        "test_size": test_size,
        "seed": seed,
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "per_class_accuracy": per_class_accuracy,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "n_test": int(len(y_test)),
    }


def evaluate_regression(test_size: float, seed: int) -> dict:
    _, _, reg_model = load_models()
    combined_base, _ = build_frames()

    regression_frame = combined_base.copy()
    regression_X = regression_frame.drop(columns=["Player"] + sorted(REGRESSION_TARGET_LEAKAGE_COLUMNS), errors="ignore")
    regression_y = regression_frame["Gls"]

    X_train, X_test, y_train, y_test = train_test_split(
        regression_X,
        regression_y,
        test_size=test_size,
        random_state=seed,
    )

    fitted_model = clone(reg_model)
    fitted_model.fit(X_train, y_train)

    train_pred = fitted_model.predict(X_train)
    y_pred = fitted_model.predict(X_test)
    cv_scores = cross_val_score(clone(reg_model), regression_X, regression_y, cv=5, scoring="r2")

    return {
        "mode": "regression",
        "model": "combined_base_regression_ElasticNet",
        "test_size": test_size,
        "seed": seed,
        "train_r2": float(r2_score(y_train, train_pred)),
        "holdout_r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "n_test": int(len(y_test)),
        "preview": [
            {
                "actual_gls": float(a),
                "pred_gls": float(np.round(p, 3)),
            }
            for a, p in zip(y_test.iloc[:10].tolist(), y_pred[:10].tolist())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime evaluation for exported football models")
    parser.add_argument("--mode", choices=["sample", "classification", "regression"], required=True)
    parser.add_argument("--dataset", choices=["combined", "detailed"], default="combined")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "sample":
        result = sample_predictions(dataset=args.dataset, n=args.n, seed=args.seed)
    elif args.mode == "classification":
        result = evaluate_classification(dataset=args.dataset, test_size=args.test_size, seed=args.seed)
    else:
        result = evaluate_regression(test_size=args.test_size, seed=args.seed)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
