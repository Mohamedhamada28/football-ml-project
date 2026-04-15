from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from football_model_refresh import (
    REGRESSION_TARGET_LEAKAGE_COLUMNS,
    make_classification_models,
    make_regression_models,
)


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "webapp"
OUTPUT_DIR = ROOT / "outputs" / "model_refresh"
MODEL_DIR = OUTPUT_DIR / "models"

COMBINED_CLASSIFIER_PATH = MODEL_DIR / "combined_base_HistGradientBoosting_classification.joblib"
GOALS_REGRESSOR_PATH = MODEL_DIR / "combined_base_regression_ElasticNet_regression.joblib"

COMBINED_TRAINING_PATH = OUTPUT_DIR / "combined_base_training_data.csv"
DETAILED_TRAINING_PATH = OUTPUT_DIR / "detailed_2024_2025_training_data.csv"
RUN_METADATA_PATH = OUTPUT_DIR / "run_metadata.json"
BASELINE_METRICS_PATH = OUTPUT_DIR / "baseline_metrics.csv"
CLASSIFICATION_SUMMARY_PATH = OUTPUT_DIR / "classification_summary.csv"
CLASSIFICATION_PER_CLASS_PATH = OUTPUT_DIR / "classification_per_class_metrics.csv"
CLASSIFICATION_CONFUSION_PATH = OUTPUT_DIR / "classification_confusion_matrices.csv"
CLASSIFICATION_FEATURES_PATH = OUTPUT_DIR / "classification_feature_views.csv"
REGRESSION_SUMMARY_PATH = OUTPUT_DIR / "regression_summary.csv"
REGRESSION_FEATURES_PATH = OUTPUT_DIR / "regression_feature_views.csv"

EXPERIMENT_LABELS = {
    "combined_base": "Shared Two-Season Base",
    "detailed_2024_2025": "Detailed 2024/25",
    "combined_base_regression": "Goals Regression",
}

POSITION_LABELS = {
    "DF": "Defender",
    "FW": "Forward",
    "GK": "Goalkeeper",
    "MF": "Midfielder",
}
VALID_POSITIONS = {"DF", "FW", "GK", "MF"}

SEASON_TO_SOURCE_FILE = {
    "2023_2024": "top5-players.csv",
    "2024_2025": "players_data-2024_2025.csv",
}

CLASSIFICATION_INPUT_FIELDS = [
    "Nation",
    "Squad",
    "Comp",
    "season",
    "Age",
    "Born",
    "MP",
    "Starts",
    "Min",
    "90s",
    "Gls",
    "Ast",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "xG",
    "npxG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]

GOALS_INPUT_FIELDS = [
    "Nation",
    "Squad",
    "Comp",
    "Pos",
    "season",
    "Age",
    "Born",
    "MP",
    "Starts",
    "Min",
    "90s",
    "Ast",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "xG",
    "npxG",
    "xAG",
    "PrgC",
    "PrgP",
    "PrgR",
]

PER90_BASE_COLUMNS = [
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

REGRESSION_PER90_COLUMNS = [
    "Ast",
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

CATEGORICAL_PREFIXES = ("Nation_", "Squad_", "Comp_", "season_", "source_file_", "Pos_")
BALLON_DOR_SCORE_WEIGHTS = {
    "Gls": 0.26,
    "G+A": 0.22,
    "xa_combo": 0.16,
    "pred_goals": 0.14,
    "Gls_per90": 0.10,
    "xAG_per90": 0.06,
    "progression_combo": 0.04,
    "start_rate": 0.02,
}


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def metric_pct(value: Any, digits: int = 1) -> float:
    return round(float(value) * 100, digits)


def friendly_feature_name(name: str) -> str:
    return (
        name.replace("_per90", " /90")
        .replace("Prg", "Prog ")
        .replace("G+A", "Goals + Assists")
        .replace("G-PK", "Non-penalty goals")
        .replace("xAG", "xAG")
        .replace("xG", "xG")
        .replace("npxG", "npxG")
        .replace("CrdY", "Yellow cards")
        .replace("CrdR", "Red cards")
        .replace("PKatt", "Penalties attempted")
        .replace("PK", "Penalty goals")
        .replace("start_rate", "Start rate")
        .replace("  ", " ")
    )


class DashboardStore:
    def __init__(self) -> None:
        self._assert_required_files()

        self.run_metadata = json.loads(RUN_METADATA_PATH.read_text(encoding="utf-8"))
        self.combined_training = pd.read_csv(COMBINED_TRAINING_PATH)
        self.detailed_training = pd.read_csv(DETAILED_TRAINING_PATH)
        self.baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
        self.classification_summary = pd.read_csv(CLASSIFICATION_SUMMARY_PATH)
        self.classification_per_class = pd.read_csv(CLASSIFICATION_PER_CLASS_PATH)
        self.classification_confusion = pd.read_csv(CLASSIFICATION_CONFUSION_PATH)
        self.classification_features = pd.read_csv(CLASSIFICATION_FEATURES_PATH)
        self.regression_summary = pd.read_csv(REGRESSION_SUMMARY_PATH)
        self.regression_features = pd.read_csv(REGRESSION_FEATURES_PATH)

        self.combined_classifier = joblib.load(COMBINED_CLASSIFIER_PATH)
        self.goals_regressor = joblib.load(GOALS_REGRESSOR_PATH)
        self.classification_feature_names = list(
            self.combined_classifier.named_steps["preprocessor"].feature_names_in_
        )
        self.regression_feature_names = list(
            self.goals_regressor.named_steps["preprocessor"].feature_names_in_
        )
        self.classification_labels = list(self.combined_classifier.named_steps["model"].classes_)
        self.goal_distribution = self.combined_training["Gls"].astype(float).to_numpy()
        self.position_model_rows = self.classification_summary[
            self.classification_summary["experiment"] == "combined_base"
        ].copy()
        self.regression_model_rows = self.regression_summary.copy()
        self.available_position_model_names = self.position_model_rows["model"].tolist()
        self.available_goals_model_names = self.regression_model_rows["model"].tolist()
        self.combined_classification_X = self.combined_training.drop(columns=["Pos", "Player"])
        self.combined_classification_y = self.combined_training["Pos"]
        self.combined_regression_X = self.combined_training.drop(
            columns=["Player"] + sorted(REGRESSION_TARGET_LEAKAGE_COLUMNS),
            errors="ignore",
        )
        self.combined_regression_y = self.combined_training["Gls"]
        self.live_position_models: dict[str, Any] = {"HistGradientBoosting": self.combined_classifier}
        self.live_goals_models: dict[str, Any] = {"ElasticNet": self.goals_regressor}
        self.latest_player_analysis = self._build_latest_player_analysis()

        self.dashboard_payload = self._build_dashboard_payload()

    def _assert_required_files(self) -> None:
        required = [
            COMBINED_CLASSIFIER_PATH,
            GOALS_REGRESSOR_PATH,
            COMBINED_TRAINING_PATH,
            DETAILED_TRAINING_PATH,
            RUN_METADATA_PATH,
            BASELINE_METRICS_PATH,
            CLASSIFICATION_SUMMARY_PATH,
            CLASSIFICATION_PER_CLASS_PATH,
            CLASSIFICATION_CONFUSION_PATH,
            CLASSIFICATION_FEATURES_PATH,
            REGRESSION_SUMMARY_PATH,
            REGRESSION_FEATURES_PATH,
            WEB_DIR / "index.html",
            WEB_DIR / "app.js",
            WEB_DIR / "styles.css",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing app assets or model outputs: {missing}")

    def _build_dashboard_payload(self) -> dict[str, Any]:
        combined_best = self._best_classification_row("combined_base")
        detailed_best = self._best_classification_row("detailed_2024_2025")
        regression_best = self.regression_summary.iloc[0]

        baseline = {
            row["metric"]: float(row["value"])
            for _, row in self.baseline_metrics.iterrows()
        }
        leakage_count = len(self.run_metadata["detailed_dropped_metadata_columns"])
        mixed_label_stats = self._mixed_label_stats()

        return {
            "hero": {
                "title": "Football Models.\nMade Visible.",
                "subtitle": "A Wise-inspired dashboard for accuracy, comparisons, findings, and live model-backed predictions.",
                "meta": {
                    "combined_rows": int(self.run_metadata["combined_base_shape"][0]),
                    "detailed_rows": int(self.run_metadata["detailed_2024_2025_shape"][0]),
                    "combined_features": len(self.classification_feature_names),
                    "detailed_features": int(self.detailed_training.drop(columns=["Player", "Pos"]).shape[1]),
                },
            },
            "kpis": [
                {
                    "label": "Shared Base Accuracy",
                    "value": metric_pct(combined_best["holdout_accuracy"]),
                    "suffix": "%",
                    "detail": f"{combined_best['model']} on the safe two-season feature set",
                    "tone": "dark",
                },
                {
                    "label": "Detailed Accuracy",
                    "value": metric_pct(detailed_best["holdout_accuracy"]),
                    "suffix": "%",
                    "detail": f"+{metric_pct(float(detailed_best['holdout_accuracy']) - float(combined_best['holdout_accuracy'])):.1f} pts from richer 2024/25 features",
                    "tone": "green",
                },
                {
                    "label": "Goals R2",
                    "value": round(float(regression_best["holdout_r2"]), 3),
                    "suffix": "",
                    "detail": f"{regression_best['model']} beats the notebook baseline by {float(regression_best['holdout_r2']) - baseline['notebook_linear_holdout_r2_unrounded']:.4f}",
                    "tone": "light",
                },
                {
                    "label": "Leakage Columns Removed",
                    "value": leakage_count,
                    "suffix": "",
                    "detail": "Duplicated detailed metadata dropped before training",
                    "tone": "light",
                },
            ],
            "classificationComparison": self._classification_chart_rows(),
            "regressionComparison": self._regression_chart_rows(),
            "baselineComparison": {
                "classification_holdout_baseline": metric_pct(baseline["notebook_rf_holdout_accuracy"]),
                "classification_cv_baseline": metric_pct(baseline["notebook_rf_cv_accuracy_mean"]),
                "combined_improvement": metric_pct(
                    float(combined_best["holdout_accuracy"]) - baseline["notebook_rf_holdout_accuracy"]
                ),
                "detailed_improvement": metric_pct(
                    float(detailed_best["holdout_accuracy"]) - baseline["notebook_rf_holdout_accuracy"]
                ),
                "regression_baseline_r2": round(baseline["notebook_linear_holdout_r2_unrounded"], 3),
                "regression_baseline_mse": round(baseline["notebook_linear_holdout_mse_unrounded"], 3),
                "regression_mse_delta": round(
                    float(regression_best["holdout_mse"]) - baseline["notebook_linear_holdout_mse_unrounded"],
                    3,
                ),
            },
            "findings": self._build_findings(combined_best, detailed_best, regression_best, mixed_label_stats),
            "classBreakdown": {
                "combined": self._best_per_class_cards("combined_base"),
                "detailed": self._best_per_class_cards("detailed_2024_2025"),
            },
            "confusionMatrices": {
                "combined": self._best_confusion_matrix("combined_base"),
                "detailed": self._best_confusion_matrix("detailed_2024_2025"),
            },
            "featureHighlights": {
                "classification": self._top_classification_features("combined_base"),
                "detailedClassification": self._top_classification_features("detailed_2024_2025"),
                "regression": self._top_regression_features(),
            },
            "modelExamples": self._build_model_examples(),
            "predictor": {
                "note": "The live predictor uses the shared-base classifier and the exported ElasticNet goals regressor. The detailed model stays in comparison mode because it needs 191 inputs.",
                "schema": self._build_schema(),
            },
            "awardRadar": self._build_award_radar(),
        }

    def _best_classification_row(self, experiment: str) -> pd.Series:
        subset = self.classification_summary[self.classification_summary["experiment"] == experiment]
        return subset.sort_values(
            ["holdout_macro_f1", "holdout_accuracy"],
            ascending=[False, False],
        ).iloc[0]

    def _classification_chart_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        ordered = self.classification_summary.copy().sort_values(
            ["experiment", "holdout_accuracy"],
            ascending=[True, False],
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "experiment": row["experiment"],
                    "experimentLabel": EXPERIMENT_LABELS.get(row["experiment"], row["experiment"]),
                    "model": row["model"],
                    "accuracy": metric_pct(row["holdout_accuracy"]),
                    "macroF1": metric_pct(row["holdout_macro_f1"]),
                    "midfielderRecall": metric_pct(row["holdout_mf_recall"]),
                    "cvAccuracy": metric_pct(row["cv_accuracy_mean"]),
                }
            )
        return rows

    def _regression_chart_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        ordered = self.regression_summary.copy().sort_values(
            ["holdout_r2", "holdout_mae"],
            ascending=[False, True],
        )
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "r2": round(float(row["holdout_r2"]), 3),
                    "mae": round(float(row["holdout_mae"]), 3),
                    "rmse": round(float(row["holdout_rmse"]), 3),
                    "mse": round(float(row["holdout_mse"]), 3),
                }
            )
        return rows

    def _mixed_label_stats(self) -> dict[str, Any]:
        top5_raw = pd.read_csv(ROOT / "top5-players.csv")
        detailed_raw = pd.read_csv(ROOT / "players_data-2024_2025.csv")

        def summarize(frame: pd.DataFrame) -> dict[str, Any]:
            filtered = frame[frame["Min"] > 90].copy()
            mixed = int(filtered["Pos"].astype(str).str.contains(",").sum())
            return {
                "rows": int(len(filtered)),
                "mixedRows": mixed,
                "mixedPct": round((mixed / len(filtered)) * 100, 1),
            }

        return {
            "top5": summarize(top5_raw),
            "detailed": summarize(detailed_raw),
        }

    def _build_latest_player_analysis(self) -> pd.DataFrame:
        latest = self.combined_training[self.combined_training["season"] == "2024_2025"].copy()
        classification_frame = latest[self.classification_feature_names].copy()
        regression_frame = latest[self.regression_feature_names].copy()

        probabilities = self.combined_classifier.predict_proba(classification_frame)
        latest["pred_pos"] = self.combined_classifier.predict(classification_frame)
        latest["pos_conf"] = probabilities.max(axis=1)
        latest["position_correct"] = latest["pred_pos"] == latest["Pos"]
        latest["pred_goals"] = np.clip(self.goals_regressor.predict(regression_frame), 0, None)
        latest["goal_error"] = (latest["Gls"] - latest["pred_goals"]).abs()
        latest["goal_delta"] = latest["Gls"] - latest["pred_goals"]
        latest["xa_combo"] = latest["xG"] + latest["xAG"]
        latest["progression_combo"] = latest["PrgP_per90"] + latest["PrgR_per90"]

        score = pd.Series(0.0, index=latest.index)
        for column, weight in BALLON_DOR_SCORE_WEIGHTS.items():
            score += latest[column].rank(pct=True) * weight
        latest["ballon_score"] = score * 100
        return latest

    def _build_model_examples(self) -> dict[str, Any]:
        latest_outfield = self.latest_player_analysis[self.latest_player_analysis["Pos"] != "GK"].copy()

        accurate_position_rows: list[pd.Series] = []
        for position in ["FW", "MF", "DF"]:
            top_rows = latest_outfield[
                (latest_outfield["Pos"] == position)
                & (latest_outfield["position_correct"])
            ].sort_values(["G+A", "pos_conf"], ascending=[False, False]).head(2)
            accurate_position_rows.extend([row for _, row in top_rows.iterrows()])

        position_misses = latest_outfield[
            ~latest_outfield["position_correct"]
        ].sort_values(["G+A", "pos_conf"], ascending=[False, False]).head(6)

        goal_hits = latest_outfield[
            (latest_outfield["goal_error"] <= 0.8) & (latest_outfield["Gls"] >= 15)
        ].sort_values(["Gls", "goal_error"], ascending=[False, True]).head(6)

        goal_misses = latest_outfield[
            (latest_outfield["goal_error"] >= 3.0)
            & ((latest_outfield["Gls"] >= 10) | (latest_outfield["pred_goals"] >= 10))
        ].sort_values(["goal_error", "Gls"], ascending=[False, False]).head(6)

        return {
            "modelsUsed": {
                "position": "HistGradientBoosting on the shared two-season base table",
                "goals": "ElasticNet on the shared two-season base regression table",
            },
            "position": {
                "accurate": [self._serialize_position_example(row) for row in accurate_position_rows[:6]],
                "misses": [self._serialize_position_example(row) for _, row in position_misses.iterrows()],
            },
            "goals": {
                "accurate": [self._serialize_goal_example(row) for _, row in goal_hits.iterrows()],
                "misses": [self._serialize_goal_example(row) for _, row in goal_misses.iterrows()],
            },
        }

    def _build_award_radar(self) -> dict[str, Any]:
        latest = self.latest_player_analysis[self.latest_player_analysis["90s"] >= 15].copy()
        candidates = latest.sort_values("ballon_score", ascending=False).head(8)

        league_leaders: list[dict[str, Any]] = []
        for league, group in latest.groupby("Comp"):
            winner = group.sort_values("ballon_score", ascending=False).iloc[0]
            league_leaders.append(
                {
                    "league": str(league),
                    "player": str(winner["Player"]),
                    "squad": str(winner["Squad"]),
                    "score": round(float(winner["ballon_score"]), 1),
                }
            )
        league_leaders = sorted(league_leaders, key=lambda item: item["score"], reverse=True)

        strong_finish = latest[latest["Gls"] >= 10].sort_values("goal_delta", ascending=False).iloc[0]
        strong_creator = latest.sort_values(["G+A", "xAG"], ascending=[False, False]).iloc[0]
        hybrid_case = latest[
            (latest["Pos"] != "GK") & (latest["G+A"] >= 10)
        ].sort_values(["pos_conf", "G+A"], ascending=[True, False]).iloc[0]

        return {
            "title": "Ballon d'Or Radar",
            "subtitle": "A dataset-driven shortlist from the 2024/25 rows. The score blends goals, total output, expected attacking value, predicted goals, per-90 punch, progression, and availability.",
            "methodology": "This is not an official award prediction. It is a model-and-stats radar built from the available player database after the refresh pipeline.",
            "modelsUsed": {
                "position": "Shared-base HistGradientBoosting classifier for position confidence and role stability.",
                "goals": "Shared-base ElasticNet regressor for expected goals output from leakage-safe inputs.",
            },
            "howItWorks": [
                "The position classifier scores how confidently each 2024/25 player still fits a stable role inside the shared-base feature space.",
                "The goals regressor estimates expected goal output without using the actual goal column as an input.",
                "The final radar score then blends actual goals, G+A, xG+xAG, predicted goals, Gls/90, xAG/90, progression, and start rate.",
            ],
            "candidates": [self._serialize_candidate(row) for _, row in candidates.iterrows()],
            "leagueLeaders": league_leaders,
            "insights": [
                {
                    "label": "Best overall production",
                    "value": strong_creator["Player"],
                    "detail": f"{int(strong_creator['G+A'])} goal involvements with {round(float(strong_creator['xAG']), 1)} xAG.",
                },
                {
                    "label": "Biggest finishing beat",
                    "value": strong_finish["Player"],
                    "detail": f"{int(strong_finish['Gls'])} actual goals versus {round(float(strong_finish['pred_goals']), 1)} predicted.",
                },
                {
                    "label": "Most hybrid role",
                    "value": hybrid_case["Player"],
                    "detail": f"Only {metric_pct(hybrid_case['pos_conf'])}% position confidence despite {int(hybrid_case['G+A'])} goal involvements.",
                },
            ],
        }

    def _serialize_candidate(self, row: pd.Series) -> dict[str, Any]:
        return {
            "player": str(row["Player"]),
            "squad": str(row["Squad"]),
            "league": str(row["Comp"]),
            "position": str(row["Pos"]),
            "positionLabel": POSITION_LABELS.get(str(row["Pos"]), str(row["Pos"])),
            "score": round(float(row["ballon_score"]), 1),
            "goals": int(row["Gls"]),
            "assists": int(row["Ast"]),
            "goalContrib": int(row["G+A"]),
            "predictedGoals": round(float(row["pred_goals"]), 1),
            "goalDelta": round(float(row["goal_delta"]), 1),
            "positionConfidence": metric_pct(row["pos_conf"]),
        }

    def _serialize_position_example(self, row: pd.Series) -> dict[str, Any]:
        return {
            "player": str(row["Player"]),
            "squad": str(row["Squad"]),
            "actual": str(row["Pos"]),
            "predicted": str(row["pred_pos"]),
            "confidence": metric_pct(row["pos_conf"]),
            "goals": int(row["Gls"]),
            "assists": int(row["Ast"]),
            "goalContrib": int(row["G+A"]),
            "progressivePasses": int(row["PrgP"]),
            "progressiveReceptions": int(row["PrgR"]),
        }

    def _serialize_goal_example(self, row: pd.Series) -> dict[str, Any]:
        return {
            "player": str(row["Player"]),
            "squad": str(row["Squad"]),
            "actualGoals": int(row["Gls"]),
            "predictedGoals": round(float(row["pred_goals"]), 1),
            "goalError": round(float(row["goal_error"]), 1),
            "position": str(row["Pos"]),
            "predictedPosition": str(row["pred_pos"]),
            "positionConfidence": metric_pct(row["pos_conf"]),
            "assists": int(row["Ast"]),
        }

    def _build_findings(
        self,
        combined_best: pd.Series,
        detailed_best: pd.Series,
        regression_best: pd.Series,
        mixed_label_stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        combined_cards = self._best_per_class_cards("combined_base")
        detailed_cards = self._best_per_class_cards("detailed_2024_2025")

        combined_mf = next(card for card in combined_cards if card["label"] == "MF")
        detailed_mf = next(card for card in detailed_cards if card["label"] == "MF")
        combined_gk = next(card for card in combined_cards if card["label"] == "GK")

        return [
            {
                "eyebrow": "Signal Depth",
                "title": "Detailed 2024/25 features add a big jump.",
                "value": f"+{metric_pct(float(detailed_best['holdout_accuracy']) - float(combined_best['holdout_accuracy'])):.1f} pts",
                "body": "The best detailed classifier reaches 91.5% holdout accuracy versus 80.0% for the safer shared-base table.",
            },
            {
                "eyebrow": "Hardest Class",
                "title": "Midfield remains the messiest role.",
                "value": f"{combined_mf['recall']}%",
                "body": f"Midfielder recall is only {combined_mf['recall']}% on the combined model, even though the detailed model lifts it to {detailed_mf['recall']}%.",
            },
            {
                "eyebrow": "Label Noise",
                "title": "Hybrid roles are everywhere in the raw labels.",
                "value": f"{mixed_label_stats['top5']['mixedPct']}% / {mixed_label_stats['detailed']['mixedPct']}%",
                "body": "About three in ten filtered rows still carry mixed positions, which makes clean tactical separation harder than the headline metrics suggest.",
            },
            {
                "eyebrow": "Reliable Class",
                "title": "Goalkeepers are almost perfectly separable.",
                "value": f"{combined_gk['recall']}%",
                "body": "GK recall is near-perfect on both experiments, which reinforces that the real classification challenge sits in adjacent outfield roles.",
            },
            {
                "eyebrow": "Goals Model",
                "title": "ElasticNet edges the notebook baseline.",
                "value": f"R2 {round(float(regression_best['holdout_r2']), 3)}",
                "body": f"MSE lands at {round(float(regression_best['holdout_mse']), 3)}, improving on the notebook linear baseline without using goal leakage features.",
            },
        ]

    def _best_per_class_cards(self, experiment: str) -> list[dict[str, Any]]:
        best_model = self._best_classification_row(experiment)["model"]
        subset = self.classification_per_class[
            (self.classification_per_class["experiment"] == experiment)
            & (self.classification_per_class["model"] == best_model)
            & (self.classification_per_class["label"].isin(["DF", "FW", "GK", "MF"]))
        ]

        cards: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            label = str(row["label"])
            cards.append(
                {
                    "label": label,
                    "fullLabel": POSITION_LABELS.get(label, label),
                    "precision": metric_pct(row["precision"]),
                    "recall": metric_pct(row["recall"]),
                    "f1": metric_pct(row["f1_score"]),
                    "support": int(float(row["support"])),
                }
            )
        return cards

    def _best_confusion_matrix(self, experiment: str) -> dict[str, Any]:
        best_model = self._best_classification_row(experiment)["model"]
        subset = self.classification_confusion[
            (self.classification_confusion["experiment"] == experiment)
            & (self.classification_confusion["model"] == best_model)
        ]
        labels = ["DF", "FW", "GK", "MF"]
        matrix: list[list[int]] = []
        for actual in labels:
            row_values: list[int] = []
            for predicted in labels:
                match = subset[
                    (subset["actual"] == actual)
                    & (subset["predicted"] == predicted)
                ]
                value = int(match.iloc[0]["count"]) if not match.empty else 0
                row_values.append(value)
            matrix.append(row_values)

        return {
            "title": EXPERIMENT_LABELS.get(experiment, experiment),
            "labels": labels,
            "matrix": matrix,
        }

    def _top_classification_features(self, experiment: str, top_n: int = 6) -> list[dict[str, Any]]:
        subset = self.classification_features[
            (self.classification_features["experiment"] == experiment)
            & (self.classification_features["model"] == "RandomForest")
            & (self.classification_features["view_type"] == "importance")
        ].copy()
        subset["value"] = subset["value"].astype(float)
        subset = subset[~subset["feature"].astype(str).str.startswith(CATEGORICAL_PREFIXES)]
        subset = subset.sort_values("value", ascending=False).head(top_n)
        return [
            {
                "feature": row["feature"],
                "label": friendly_feature_name(str(row["feature"])),
                "value": round(float(row["value"]), 3),
            }
            for _, row in subset.iterrows()
        ]

    def _top_regression_features(self, top_n: int = 6) -> list[dict[str, Any]]:
        subset = self.regression_features[
            (self.regression_features["model"] == "ElasticNet")
            & (self.regression_features["view_type"] == "coefficient")
        ].copy()
        subset["value"] = subset["value"].astype(float)
        subset = subset[~subset["feature"].astype(str).str.startswith(CATEGORICAL_PREFIXES)]
        subset = subset.sort_values("value", key=lambda series: series.abs(), ascending=False).head(top_n)
        return [
            {
                "feature": row["feature"],
                "label": friendly_feature_name(str(row["feature"])),
                "value": round(float(row["value"]), 3),
                "direction": "up" if float(row["value"]) >= 0 else "down",
            }
            for _, row in subset.iterrows()
        ]

    def _build_schema(self) -> dict[str, Any]:
        common_defaults = {
            field: self._default_for_field(field)
            for field in sorted(set(CLASSIFICATION_INPUT_FIELDS + GOALS_INPUT_FIELDS))
        }

        return {
            "classificationFields": CLASSIFICATION_INPUT_FIELDS,
            "goalsFields": GOALS_INPUT_FIELDS,
            "options": {
                "Nation": self._options_for_field("Nation"),
                "Squad": self._options_for_field("Squad"),
                "Comp": self._options_for_field("Comp"),
                "season": self._options_for_field("season"),
                "Pos": ["DF", "FW", "GK", "MF"],
            },
            "defaults": common_defaults,
            "seasonToSourceFile": SEASON_TO_SOURCE_FILE,
            "models": {
                "position": self._position_model_options(),
                "goals": self._goals_model_options(),
                "defaultPositionModel": "HistGradientBoosting",
                "defaultGoalsModel": "ElasticNet",
            },
            "inputNotes": {
                "classification": "The position classifier benchmarks same-season stat lines and auto-derives G+A, G-PK, npxG+xAG, per-90 rates, and start rate.",
                "goals": "The goals model excludes Gls from inputs. It uses position plus creation, penalty, progression, and playing-time features.",
            },
        }

    def _position_model_options(self) -> list[dict[str, Any]]:
        rows = self.position_model_rows.sort_values("holdout_accuracy", ascending=False)
        return [
            {
                "id": str(row["model"]),
                "label": str(row["model"]),
                "dataset": "Shared Two-Season Base",
                "description": f"Holdout accuracy {metric_pct(row['holdout_accuracy'])}%, macro-F1 {metric_pct(row['holdout_macro_f1'])}%, MF recall {metric_pct(row['holdout_mf_recall'])}%.",
            }
            for _, row in rows.iterrows()
        ]

    def _goals_model_options(self) -> list[dict[str, Any]]:
        rows = self.regression_model_rows.sort_values("holdout_r2", ascending=False)
        return [
            {
                "id": str(row["model"]),
                "label": str(row["model"]),
                "dataset": "Shared Two-Season Base",
                "description": f"Holdout R2 {round(float(row['holdout_r2']), 3)}, MAE {round(float(row['holdout_mae']), 3)}, RMSE {round(float(row['holdout_rmse']), 3)}.",
            }
            for _, row in rows.iterrows()
        ]

    def _get_live_position_model(self, model_name: str) -> Any:
        if model_name not in self.available_position_model_names:
            raise ValueError(f"Unsupported position model: {model_name}")
        if model_name not in self.live_position_models:
            model = make_classification_models(self.combined_classification_X)[model_name]
            model.fit(self.combined_classification_X, self.combined_classification_y)
            self.live_position_models[model_name] = model
        return self.live_position_models[model_name]

    def _get_live_goals_model(self, model_name: str) -> Any:
        if model_name not in self.available_goals_model_names:
            raise ValueError(f"Unsupported goals model: {model_name}")
        if model_name not in self.live_goals_models:
            model = make_regression_models(self.combined_regression_X)[model_name]
            model.fit(self.combined_regression_X, self.combined_regression_y)
            self.live_goals_models[model_name] = model
        return self.live_goals_models[model_name]

    def _default_for_field(self, field: str) -> Any:
        series = self.combined_training[field]
        if pd.api.types.is_numeric_dtype(series):
            value = float(series.median())
            return round(value, 2)
        return str(series.mode().iloc[0])

    def _options_for_field(self, field: str) -> list[str]:
        series = self.combined_training[field].dropna().astype(str)
        values = sorted(series.unique().tolist())
        return values

    def get_dashboard_payload(self) -> dict[str, Any]:
        return self.dashboard_payload

    def predict_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = str(payload.get("model") or "HistGradientBoosting")
        model = self._get_live_position_model(model_name)
        frame, warnings = self._build_position_frame(payload)
        predicted_label = str(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        ranked = sorted(
            (
                {
                    "label": label,
                    "fullLabel": POSITION_LABELS.get(label, label),
                    "probability": round(float(probability), 4),
                    "probabilityPct": round(float(probability) * 100, 1),
                }
                for label, probability in zip(self.classification_labels, probabilities)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "model": model_name,
            "modelDescription": next(
                option["description"]
                for option in self._position_model_options()
                if option["id"] == model_name
            ),
            "prediction": predicted_label,
            "predictionLabel": POSITION_LABELS.get(predicted_label, predicted_label),
            "topProbabilityPct": ranked[0]["probabilityPct"],
            "probabilities": ranked,
            "derived": {
                "goalsPlusAssists": round(float(frame.iloc[0]["G+A"]), 2),
                "nonPenaltyGoals": round(float(frame.iloc[0]["G-PK"]), 2),
                "npxGPlusxAG": round(float(frame.iloc[0]["npxG+xAG"]), 2),
                "goalsPer90": round(float(frame.iloc[0]["Gls_per90"]), 3),
                "xGPer90": round(float(frame.iloc[0]["xG_per90"]), 3),
                "startRate": round(float(frame.iloc[0]["start_rate"]), 3),
            },
            "warnings": warnings,
        }

    def predict_goals(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = str(payload.get("model") or "ElasticNet")
        model = self._get_live_goals_model(model_name)
        frame, warnings = self._build_goals_frame(payload)
        raw_prediction = float(model.predict(frame)[0])
        clipped_prediction = max(raw_prediction, 0.0)
        percentile = round(float((self.goal_distribution <= clipped_prediction).mean() * 100), 1)

        return {
            "model": model_name,
            "modelDescription": next(
                option["description"]
                for option in self._goals_model_options()
                if option["id"] == model_name
            ),
            "predictedGoals": round(clipped_prediction, 2),
            "rawPrediction": round(raw_prediction, 3),
            "percentile": percentile,
            "derived": {
                "xGPer90": round(float(frame.iloc[0]["xG_per90"]), 3),
                "xAGPer90": round(float(frame.iloc[0]["xAG_per90"]), 3),
                "progressivePassesPer90": round(float(frame.iloc[0]["PrgP_per90"]), 3),
                "startRate": round(float(frame.iloc[0]["start_rate"]), 3),
            },
            "warnings": warnings,
        }

    def predict_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        csv_text = str(payload.get("csvText") or "")
        if not csv_text.strip():
            raise ValueError("Upload a CSV file before running the batch predictor.")

        try:
            raw_frame = pd.read_csv(StringIO(csv_text))
        except Exception as exc:
            raise ValueError(f"Could not read the uploaded CSV: {exc}") from exc

        if raw_frame.empty:
            raise ValueError("The uploaded CSV does not contain any rows.")

        raw_frame = raw_frame.copy()
        raw_frame.columns = [str(column).strip() for column in raw_frame.columns]

        filename = str(payload.get("filename") or "uploaded_dataset.csv").strip() or "uploaded_dataset.csv"
        position_model_name = str(payload.get("positionModel") or "HistGradientBoosting")
        goals_model_name = str(payload.get("goalsModel") or "ElasticNet")
        use_predicted_position = bool(payload.get("usePredictedPosition", True))

        position_model = self._get_live_position_model(position_model_name)
        goals_model = self._get_live_goals_model(goals_model_name)

        prepared, filled_columns = self._prepare_dataset_base_frame(raw_frame, filename)
        classification_frame = self._build_position_dataset_frame(prepared)
        predicted_positions = position_model.predict(classification_frame)
        probabilities = position_model.predict_proba(classification_frame)
        position_confidence = probabilities.max(axis=1)

        fallback_positions = 0
        if use_predicted_position:
            goals_positions = pd.Series(predicted_positions, index=prepared.index)
        else:
            goals_positions, fallback_positions = self._resolve_dataset_positions(prepared["Pos"])

        regression_frame = self._build_goals_dataset_frame(prepared, goals_positions)
        predicted_goals = np.clip(goals_model.predict(regression_frame), 0, None)

        results = prepared.copy()
        results["row_number"] = np.arange(1, len(results) + 1)
        results["Player"] = self._resolve_player_labels(results["Player"], results["row_number"])
        results["predicted_position"] = predicted_positions
        results["predicted_position_label"] = results["predicted_position"].map(POSITION_LABELS).fillna(results["predicted_position"])
        results["position_confidence"] = position_confidence
        results["predicted_goals"] = predicted_goals
        results["G+A"] = results["Gls"] + results["Ast"]
        results["xa_combo"] = results["xG"] + results["xAG"]
        results["pred_goals"] = predicted_goals
        results["npxG+xAG"] = results["npxG"] + results["xAG"]
        results["Gls_per90"] = self._rate_series(results["Gls"], results["90s"])
        results["xAG_per90"] = self._rate_series(results["xAG"], results["90s"])
        results["PrgP_per90"] = self._rate_series(results["PrgP"], results["90s"])
        results["PrgR_per90"] = self._rate_series(results["PrgR"], results["90s"])
        results["progression_combo"] = results["PrgP_per90"] + results["PrgR_per90"]
        results["start_rate"] = self._rate_series(results["Starts"], results["MP"])

        score = pd.Series(0.0, index=results.index)
        for column, weight in BALLON_DOR_SCORE_WEIGHTS.items():
            score += results[column].rank(pct=True, method="average") * weight
        results["ballon_score"] = score * 100

        if "Gls" in raw_frame.columns:
            results["actual_goals"] = pd.to_numeric(raw_frame["Gls"], errors="coerce")
        else:
            results["actual_goals"] = np.nan

        low_minute_rows = int((prepared["Min"] <= 90).sum())
        warnings: list[str] = []
        if filled_columns:
            warnings.append(
                "The upload was missing some model inputs, so training defaults were used for: "
                + ", ".join(filled_columns)
                + "."
            )
        if low_minute_rows:
            warnings.append(
                f"{low_minute_rows} uploaded rows are at 90 minutes or fewer, so those predictions are less reliable."
            )
        if use_predicted_position:
            warnings.append(
                "Goals predictions used the classifier output as the position input for every uploaded row."
            )
        elif fallback_positions:
            warnings.append(
                f"{fallback_positions} rows had missing or unknown positions, so the goals model fell back to the default training position."
            )

        top_players_frame = results.sort_values(
            ["predicted_goals", "position_confidence"],
            ascending=[False, False],
        ).head(10)
        top_players = [
            self._serialize_dataset_prediction(row, rank=index + 1)
            for index, (_, row) in enumerate(top_players_frame.iterrows())
        ]
        ballon_candidates_frame = results.sort_values(
            ["ballon_score", "predicted_goals", "position_confidence"],
            ascending=[False, False, False],
        ).head(10)
        ballon_candidates = [
            self._serialize_dataset_candidate(row, rank=index + 1)
            for index, (_, row) in enumerate(ballon_candidates_frame.iterrows())
        ]

        league_top5: list[dict[str, Any]] = []
        grouped = results.sort_values(
            ["predicted_goals", "position_confidence"],
            ascending=[False, False],
        ).groupby("Comp", dropna=False)
        for league, group in grouped:
            league_name = str(league).strip() or "Unknown League"
            top_rows = group.head(5)
            league_top5.append(
                {
                    "league": league_name,
                    "players": [
                        self._serialize_dataset_prediction(row, rank=index + 1)
                        for index, (_, row) in enumerate(top_rows.iterrows())
                    ],
                }
            )
        league_top5.sort(key=lambda x: x["league"])

        preview_rows = [
            self._serialize_dataset_prediction(row)
            for _, row in results.head(12).iterrows()
        ]

        position_breakdown = (
            results["predicted_position"]
            .value_counts()
            .reindex(["FW", "MF", "DF", "GK"], fill_value=0)
        )

        return {
            "filename": filename,
            "rowsProcessed": int(len(results)),
            "columnsDetected": int(len(raw_frame.columns)),
            "positionModel": position_model_name,
            "goalsModel": goals_model_name,
            "usePredictedPosition": use_predicted_position,
            "filledColumns": filled_columns,
            "lowMinuteRows": low_minute_rows,
            "warnings": warnings,
            "summary": {
                "averagePredictedGoals": round(float(results["predicted_goals"].mean()), 2),
                "maxPredictedGoals": round(float(results["predicted_goals"].max()), 2),
                "meanPositionConfidence": round(float(results["position_confidence"].mean()) * 100, 1),
                "positionBreakdown": [
                    {
                        "label": POSITION_LABELS.get(label, label),
                        "count": int(count),
                    }
                    for label, count in position_breakdown.items()
                ],
            },
            "ballonCandidates": ballon_candidates,
            "topLeaguePlayers": league_top5,
            "topPlayers": top_players,
        }

    def _prepare_dataset_base_frame(
        self,
        raw_frame: pd.DataFrame,
        filename: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        prepared = raw_frame.copy()
        prepared.columns = [str(column).strip() for column in prepared.columns]

        if "Nation" in prepared.columns:
            prepared["Nation"] = prepared["Nation"].map(self._normalize_prefixed_text).replace("", pd.NA)
        if "Comp" in prepared.columns:
            prepared["Comp"] = prepared["Comp"].map(self._normalize_prefixed_text).replace("", pd.NA)
        if "Pos" in prepared.columns:
            prepared["Pos"] = (
                prepared["Pos"]
                .fillna("")
                .astype(str)
                .str.split(",", n=1)
                .str[0]
                .str.strip()
                .replace("", pd.NA)
            )

        filled_columns: list[str] = []
        categorical_defaults = {
            "Nation": str(self._default_for_field("Nation")),
            "Squad": str(self._default_for_field("Squad")),
            "Comp": str(self._default_for_field("Comp")),
            "Pos": str(self._default_for_field("Pos")),
            "season": str(self._default_for_field("season")),
        }
        for field, default in categorical_defaults.items():
            if field not in prepared.columns:
                prepared[field] = default
                filled_columns.append(field)
                continue

            values = prepared[field].fillna("").astype(str).str.strip()
            if field == "Pos":
                values = values.str.upper()
            prepared[field] = values.replace("", default)

        if "source_file" not in prepared.columns:
            prepared["source_file"] = prepared["season"].map(SEASON_TO_SOURCE_FILE).fillna(filename)
            filled_columns.append("source_file")
        else:
            source_defaults = prepared["season"].map(SEASON_TO_SOURCE_FILE).fillna(filename)
            source_values = prepared["source_file"].fillna("").astype(str).str.strip()
            prepared["source_file"] = source_values.where(source_values != "", source_defaults)

        if "Player" not in prepared.columns:
            prepared["Player"] = ""

        numeric_fields = [
            "Age",
            "Born",
            "MP",
            "Starts",
            "Min",
            "90s",
            "Gls",
            "Ast",
            "PK",
            "PKatt",
            "CrdY",
            "CrdR",
            "xG",
            "npxG",
            "xAG",
            "PrgC",
            "PrgP",
            "PrgR",
        ]
        for field in numeric_fields:
            default = float(self._default_for_field(field))
            if field not in prepared.columns:
                prepared[field] = default
                filled_columns.append(field)
                continue
            prepared[field] = pd.to_numeric(prepared[field], errors="coerce").fillna(default)

        prepared["Player"] = prepared["Player"].fillna("").astype(str).str.strip()
        return prepared, sorted(set(filled_columns))

    def _build_position_dataset_frame(self, prepared: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=prepared.index)
        frame["Nation"] = prepared["Nation"]
        frame["Squad"] = prepared["Squad"]
        frame["Comp"] = prepared["Comp"]
        frame["season"] = prepared["season"]
        frame["source_file"] = prepared["source_file"]
        frame["Age"] = prepared["Age"]
        frame["Born"] = prepared["Born"]
        frame["MP"] = prepared["MP"]
        frame["Starts"] = prepared["Starts"]
        frame["Min"] = prepared["Min"]
        frame["90s"] = prepared["90s"]
        frame["Gls"] = prepared["Gls"]
        frame["Ast"] = prepared["Ast"]
        frame["G+A"] = prepared["Gls"] + prepared["Ast"]
        frame["G-PK"] = prepared["Gls"] - prepared["PK"]
        frame["PK"] = prepared["PK"]
        frame["PKatt"] = prepared["PKatt"]
        frame["CrdY"] = prepared["CrdY"]
        frame["CrdR"] = prepared["CrdR"]
        frame["xG"] = prepared["xG"]
        frame["npxG"] = prepared["npxG"]
        frame["xAG"] = prepared["xAG"]
        frame["npxG+xAG"] = prepared["npxG"] + prepared["xAG"]
        frame["PrgC"] = prepared["PrgC"]
        frame["PrgP"] = prepared["PrgP"]
        frame["PrgR"] = prepared["PrgR"]

        for column in PER90_BASE_COLUMNS:
            frame[f"{column}_per90"] = self._rate_series(frame[column], prepared["90s"])
        frame["start_rate"] = self._rate_series(prepared["Starts"], prepared["MP"])

        return frame[self.classification_feature_names]

    def _build_goals_dataset_frame(
        self,
        prepared: pd.DataFrame,
        positions: pd.Series,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(index=prepared.index)
        frame["Nation"] = prepared["Nation"]
        frame["Squad"] = prepared["Squad"]
        frame["Comp"] = prepared["Comp"]
        frame["Pos"] = positions
        frame["season"] = prepared["season"]
        frame["source_file"] = prepared["source_file"]
        frame["Age"] = prepared["Age"]
        frame["Born"] = prepared["Born"]
        frame["MP"] = prepared["MP"]
        frame["Starts"] = prepared["Starts"]
        frame["Min"] = prepared["Min"]
        frame["90s"] = prepared["90s"]
        frame["Ast"] = prepared["Ast"]
        frame["PK"] = prepared["PK"]
        frame["PKatt"] = prepared["PKatt"]
        frame["CrdY"] = prepared["CrdY"]
        frame["CrdR"] = prepared["CrdR"]
        frame["xG"] = prepared["xG"]
        frame["npxG"] = prepared["npxG"]
        frame["xAG"] = prepared["xAG"]
        frame["npxG+xAG"] = prepared["npxG"] + prepared["xAG"]
        frame["PrgC"] = prepared["PrgC"]
        frame["PrgP"] = prepared["PrgP"]
        frame["PrgR"] = prepared["PrgR"]

        for column in REGRESSION_PER90_COLUMNS:
            frame[f"{column}_per90"] = self._rate_series(frame[column], prepared["90s"])
        frame["start_rate"] = self._rate_series(prepared["Starts"], prepared["MP"])

        return frame[self.regression_feature_names]

    def _resolve_dataset_positions(self, raw_positions: pd.Series) -> tuple[pd.Series, int]:
        normalized = (
            raw_positions
            .fillna("")
            .astype(str)
            .str.split(",", n=1)
            .str[0]
            .str.strip()
            .str.upper()
        )
        valid_mask = normalized.isin(VALID_POSITIONS)
        fallback = str(self._default_for_field("Pos"))
        return normalized.where(valid_mask, fallback), int((~valid_mask).sum())

    def _resolve_player_labels(self, players: pd.Series, row_numbers: pd.Series) -> pd.Series:
        labels = players.fillna("").astype(str).str.strip()
        fallback = row_numbers.map(lambda value: f"Row {int(value)}")
        return labels.where(labels != "", fallback)

    def _normalize_prefixed_text(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        parts = text.split(" ", 1)
        if len(parts) == 2:
            prefix = parts[0]
            if len(prefix) <= 3 and prefix.isalpha() and (prefix.islower() or prefix.isupper()):
                return parts[1].strip()
        return text

    def _serialize_dataset_prediction(
        self,
        row: pd.Series,
        rank: int | None = None,
    ) -> dict[str, Any]:
        predicted_position = str(row["predicted_position"])
        return {
            "rank": int(rank or row["row_number"]),
            "player": str(row["Player"]),
            "squad": str(row["Squad"]),
            "league": str(row["Comp"]),
            "season": str(row["season"]),
            "predictedPosition": predicted_position,
            "predictedPositionLabel": POSITION_LABELS.get(predicted_position, predicted_position),
            "positionConfidencePct": round(float(row["position_confidence"]) * 100, 1),
            "predictedGoals": round(float(row["predicted_goals"]), 2),
            "actualGoals": self._format_optional_number(row["actual_goals"]),
        }

    def _serialize_dataset_candidate(
        self,
        row: pd.Series,
        rank: int | None = None,
    ) -> dict[str, Any]:
        payload = self._serialize_dataset_prediction(row, rank)
        payload["ballonScore"] = round(float(row["ballon_score"]), 1)
        return payload

    def _format_optional_number(self, value: Any) -> str:
        if pd.isna(value):
            return "N/A"
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(round(numeric, 2))

    def _rate_series(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator_values = denominator.astype(float).replace(0, np.nan)
        values = numerator.astype(float).divide(denominator_values)
        return values.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _build_position_frame(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        values = self._normalize_common_inputs(payload)
        gls = values["Gls"]
        ast = values["Ast"]
        pk = values["PK"]
        pkatt = values["PKatt"]
        ninety = values["90s"]
        starts = values["Starts"]
        mp = values["MP"]
        npxg = values["npxG"]
        xag = values["xAG"]

        warnings: list[str] = []
        if values["Min"] <= 90:
            warnings.append("Training rows were filtered to players above 90 minutes, so low-minute inputs are less reliable.")
        if pk > gls:
            warnings.append("Penalty goals are higher than total goals. The model will still run, but the profile is inconsistent.")
        if pk > pkatt:
            warnings.append("Penalty goals are higher than penalties attempted. Double-check PK and PKatt.")

        row = {
            "Nation": values["Nation"],
            "Squad": values["Squad"],
            "Comp": values["Comp"],
            "season": values["season"],
            "source_file": SEASON_TO_SOURCE_FILE.get(values["season"], values["source_file"]),
            "Age": values["Age"],
            "Born": values["Born"],
            "MP": mp,
            "Starts": starts,
            "Min": values["Min"],
            "90s": ninety,
            "Gls": gls,
            "Ast": ast,
            "G+A": gls + ast,
            "G-PK": gls - pk,
            "PK": pk,
            "PKatt": pkatt,
            "CrdY": values["CrdY"],
            "CrdR": values["CrdR"],
            "xG": values["xG"],
            "npxG": npxg,
            "xAG": xag,
            "npxG+xAG": npxg + xag,
            "PrgC": values["PrgC"],
            "PrgP": values["PrgP"],
            "PrgR": values["PrgR"],
        }
        for column in PER90_BASE_COLUMNS:
            row[f"{column}_per90"] = safe_rate(float(row[column]), ninety)
        row["start_rate"] = safe_rate(starts, mp)

        frame = pd.DataFrame([row], columns=self.classification_feature_names)
        return frame, warnings

    def _build_goals_frame(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        values = self._normalize_common_inputs(payload)
        ninety = values["90s"]
        starts = values["Starts"]
        mp = values["MP"]
        npxg = values["npxG"]
        xag = values["xAG"]

        warnings: list[str] = []
        if values["Min"] <= 90:
            warnings.append("Training rows were filtered to players above 90 minutes, so low-minute inputs are less reliable.")
        if values["PK"] > values["PKatt"]:
            warnings.append("Penalty goals are higher than penalties attempted. Double-check PK and PKatt.")

        position = str(payload.get("Pos") or self._default_for_field("Pos")).strip().upper()
        if position not in {"DF", "FW", "GK", "MF"}:
            position = str(self._default_for_field("Pos"))
            warnings.append("Unknown position supplied, so the goals model fell back to the default training-mode position.")

        row = {
            "Nation": values["Nation"],
            "Squad": values["Squad"],
            "Comp": values["Comp"],
            "Pos": position,
            "season": values["season"],
            "source_file": SEASON_TO_SOURCE_FILE.get(values["season"], values["source_file"]),
            "Age": values["Age"],
            "Born": values["Born"],
            "MP": mp,
            "Starts": starts,
            "Min": values["Min"],
            "90s": ninety,
            "Ast": values["Ast"],
            "PK": values["PK"],
            "PKatt": values["PKatt"],
            "CrdY": values["CrdY"],
            "CrdR": values["CrdR"],
            "xG": values["xG"],
            "npxG": npxg,
            "xAG": xag,
            "npxG+xAG": npxg + xag,
            "PrgC": values["PrgC"],
            "PrgP": values["PrgP"],
            "PrgR": values["PrgR"],
        }
        for column in REGRESSION_PER90_COLUMNS:
            row[f"{column}_per90"] = safe_rate(float(row[column]), ninety)
        row["start_rate"] = safe_rate(starts, mp)

        frame = pd.DataFrame([row], columns=self.regression_feature_names)
        return frame, warnings

    def _normalize_common_inputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        season = str(payload.get("season") or self._default_for_field("season"))
        source_file = str(
            payload.get("source_file")
            or SEASON_TO_SOURCE_FILE.get(season, self._default_for_field("source_file"))
        )

        result: dict[str, Any] = {
            "Nation": str(payload.get("Nation") or self._default_for_field("Nation")).strip(),
            "Squad": str(payload.get("Squad") or self._default_for_field("Squad")).strip(),
            "Comp": str(payload.get("Comp") or self._default_for_field("Comp")).strip(),
            "season": season,
            "source_file": source_file,
        }

        numeric_defaults = {
            field: float(self._default_for_field(field))
            for field in [
                "Age",
                "Born",
                "MP",
                "Starts",
                "Min",
                "90s",
                "Gls",
                "Ast",
                "PK",
                "PKatt",
                "CrdY",
                "CrdR",
                "xG",
                "npxG",
                "xAG",
                "PrgC",
                "PrgP",
                "PrgR",
            ]
        }
        for field, default in numeric_defaults.items():
            raw = payload.get(field, default)
            try:
                result[field] = float(raw)
            except (TypeError, ValueError):
                result[field] = default

        return result


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    store: DashboardStore

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dashboard":
            self._send_json(HTTPStatus.OK, self.store.get_dashboard_payload())
            return
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload."})
            return

        try:
            if parsed.path == "/api/predict/position":
                response = self.store.predict_position(payload)
                self._send_json(HTTPStatus.OK, response)
                return
            if parsed.path == "/api/predict/goals":
                response = self.store.predict_goals(payload)
                self._send_json(HTTPStatus.OK, response)
                return
            if parsed.path == "/api/predict/dataset":
                response = self.store.predict_dataset(payload)
                self._send_json(HTTPStatus.OK, response)
                return
        except Exception as exc:  # pragma: no cover
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(port: int) -> None:
    store = DashboardStore()
    handler = type(
        "BoundDashboardHandler",
        (DashboardRequestHandler,),
        {"store": store},
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Dashboard available at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the football model dashboard.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)


if __name__ == "__main__":
    main()
