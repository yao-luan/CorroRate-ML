from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class ExperimentConfig:
    data: Path
    sheet: str
    target: str
    output_dir: Path
    top_k: int
    test_size: float
    random_state: int
    cv: int
    skip_shap: bool


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run a reproducible corrosion-rate prediction baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/project_target_dataset.xlsx"),
        help="Path to the Excel dataset.",
    )
    parser.add_argument("--sheet", default="10year", help="Excel sheet name.")
    parser.add_argument("--target", default="RATE", help="Target column name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/corrosion_baseline"),
        help="Directory for generated metrics and figures.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of selected features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip optional SHAP plots even when shap is installed.",
    )
    args = parser.parse_args(argv)
    return ExperimentConfig(**vars(args))


def load_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.Series]:
    if not config.data.exists():
        raise FileNotFoundError(f"Dataset not found: {config.data}")

    df = pd.read_excel(config.data, sheet_name=config.sheet, engine="openpyxl")
    if config.target not in df.columns:
        raise ValueError(f"Target column {config.target!r} not found in {config.data}")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if config.target not in numeric_df.columns:
        raise ValueError(f"Target column {config.target!r} must be numeric.")

    numeric_df = numeric_df.dropna(axis=0)
    y = numeric_df[config.target].copy()
    X = numeric_df.drop(columns=[config.target]).copy()
    if X.empty:
        raise ValueError("No numeric predictor columns found.")
    return X, y


def split_and_scale(
    X: pd.DataFrame, y: pd.Series, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        x_scaler.fit_transform(X_train_raw),
        columns=X.columns,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        x_scaler.transform(X_test_raw),
        columns=X.columns,
        index=X_test_raw.index,
    )
    y_train = y_scaler.fit_transform(y_train_raw.to_numpy().reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test_raw.to_numpy().reshape(-1, 1)).ravel()
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_raw.to_numpy(),
        y_test_raw.to_numpy(),
        y_scaler,
    )


def select_features(
    X_train: pd.DataFrame, y_train: np.ndarray, config: ExperimentConfig
) -> tuple[list[str], pd.DataFrame]:
    top_k = min(config.top_k, X_train.shape[1])

    mi_scores = mutual_info_regression(
        X_train,
        y_train,
        random_state=config.random_state,
    )
    rf_probe = RandomForestRegressor(
        n_estimators=300,
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf_probe.fit(X_train, y_train)

    feature_scores = pd.DataFrame(
        {
            "feature": X_train.columns,
            "mutual_information": mi_scores,
            "random_forest_importance": rf_probe.feature_importances_,
        }
    )
    feature_scores["mi_rank"] = feature_scores["mutual_information"].rank(
        method="first", ascending=False
    )
    feature_scores["rf_rank"] = feature_scores["random_forest_importance"].rank(
        method="first", ascending=False
    )
    feature_scores["mean_rank"] = feature_scores[["mi_rank", "rf_rank"]].mean(axis=1)
    feature_scores = feature_scores.sort_values(["mean_rank", "mi_rank"])

    mi_top = (
        feature_scores.sort_values("mi_rank").head(top_k)["feature"].tolist()
    )
    rf_top = (
        feature_scores.sort_values("rf_rank").head(top_k)["feature"].tolist()
    )

    selected = [feature for feature in mi_top if feature in rf_top]
    for feature in mi_top + rf_top + feature_scores["feature"].tolist():
        if feature not in selected:
            selected.append(feature)
        if len(selected) == top_k:
            break

    feature_scores["selected"] = feature_scores["feature"].isin(selected)
    feature_scores["selected_rank"] = feature_scores["feature"].map(
        {feature: rank + 1 for rank, feature in enumerate(selected)}
    )
    return selected, feature_scores


def fit_models(
    X_train: pd.DataFrame, y_train: np.ndarray, config: ExperimentConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    cv = min(config.cv, len(y_train))
    if cv < 2:
        raise ValueError("At least two training samples are required for cross-validation.")

    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=config.random_state),
        param_grid=rf_param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)

    alphas = np.logspace(-5, 5, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring="r2")
    ridge_cv.fit(X_train, y_train)
    ridge = Ridge(alpha=float(ridge_cv.alpha_))
    ridge.fit(X_train, y_train)

    models = {
        "RandomForest": rf_grid.best_estimator_,
        "Ridge": ridge,
    }
    params = {
        "RandomForest": rf_grid.best_params_,
        "Ridge": {"alpha": float(ridge_cv.alpha_)},
    }
    return models, params


def inverse_target(y_scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def evaluate_models(
    models: dict[str, Any],
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    y_scaler: MinMaxScaler,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    prediction_frame = pd.DataFrame({"y_true": y_test_raw})

    for name, model in models.items():
        train_pred = inverse_target(y_scaler, model.predict(X_train))
        test_pred = inverse_target(y_scaler, model.predict(X_test))
        prediction_frame[f"{name}_pred"] = test_pred

        rows.append(
            {
                "model": name,
                "train_rmse": float(np.sqrt(mean_squared_error(y_train_raw, train_pred))),
                "train_mae": float(mean_absolute_error(y_train_raw, train_pred)),
                "train_r2": float(r2_score(y_train_raw, train_pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test_raw, test_pred))),
                "test_mae": float(mean_absolute_error(y_test_raw, test_pred)),
                "test_r2": float(r2_score(y_test_raw, test_pred)),
                "params": json.dumps(params[name], ensure_ascii=False),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("test_r2", ascending=False)
    best_model_name = str(metrics.iloc[0]["model"])
    metrics["best_model"] = metrics["model"].eq(best_model_name)
    return metrics, prediction_frame, best_model_name


def plot_performance(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    model_names = [column.removesuffix("_pred") for column in predictions.columns if column.endswith("_pred")]
    fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    y_true = predictions["y_true"].to_numpy()
    min_value = min(y_true.min(), *(predictions[f"{model}_pred"].min() for model in model_names))
    max_value = max(y_true.max(), *(predictions[f"{model}_pred"].max() for model in model_names))
    padding = (max_value - min_value) * 0.05 if max_value > min_value else 0.001
    limits = [min_value - padding, max_value + padding]

    for ax, model in zip(axes, model_names):
        y_pred = predictions[f"{model}_pred"]
        test_r2 = metrics.loc[metrics["model"].eq(model), "test_r2"].iloc[0]
        test_rmse = metrics.loc[metrics["model"].eq(model), "test_rmse"].iloc[0]
        ax.scatter(y_true, y_pred, alpha=0.78, edgecolor="white", linewidth=0.7)
        ax.plot(limits, limits, "--", color="black", linewidth=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(f"{model}: R2={test_r2:.3f}, RMSE={test_rmse:.4f}")
        ax.set_xlabel("Observed corrosion rate")
        ax.set_ylabel("Predicted corrosion rate")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_best_model_features(
    model: Any,
    model_name: str,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        value_name = "importance"
        title = f"{model_name} feature importance"
    elif hasattr(model, "coef_"):
        values = model.coef_
        value_name = "coefficient"
        title = f"{model_name} coefficients"
    else:
        return

    data = pd.DataFrame({"feature": feature_names, value_name: values})
    data = data.reindex(data[value_name].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x=value_name, y="feature", ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def try_plot_shap(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    output_dir: Path,
    skip_shap: bool,
) -> None:
    if skip_shap:
        return

    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency guard
        warnings.warn(f"SHAP is unavailable: {exc}")
        return

    try:
        if model_name == "RandomForest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            background = shap.sample(X_test, min(50, len(X_test)), random_state=0)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test, nsamples=100)

        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=220, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_bar.png", dpi=220, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # pragma: no cover - SHAP can be version-sensitive
        warnings.warn(f"SHAP plotting skipped: {exc}")


def save_outputs(
    config: ExperimentConfig,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_scores: pd.DataFrame,
    selected_features: list[str],
    models: dict[str, Any],
    best_model_name: str,
    X_test_selected: pd.DataFrame,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(config.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    feature_scores.to_csv(
        config.output_dir / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    serializable_config = asdict(config)
    serializable_config["data"] = str(config.data)
    serializable_config["output_dir"] = str(config.output_dir)
    serializable_config["selected_features"] = selected_features
    (config.output_dir / "run_config.json").write_text(
        json.dumps(serializable_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_performance(predictions, metrics, config.output_dir / "performance_scatter.png")
    plot_best_model_features(
        models[best_model_name],
        best_model_name,
        selected_features,
        config.output_dir / "best_model_features.png",
    )
    try_plot_shap(
        models[best_model_name],
        best_model_name,
        X_test_selected,
        config.output_dir,
        config.skip_shap,
    )


def run(config: ExperimentConfig) -> pd.DataFrame:
    X, y = load_dataset(config)
    (
        X_train,
        X_test,
        y_train,
        _y_test_scaled,
        y_train_raw,
        y_test_raw,
        y_scaler,
    ) = split_and_scale(X, y, config)

    selected_features, feature_scores = select_features(X_train, y_train, config)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    models, params = fit_models(X_train_selected, y_train, config)
    metrics, predictions, best_model_name = evaluate_models(
        models,
        params,
        X_train_selected,
        X_test_selected,
        y_train_raw,
        y_test_raw,
        y_scaler,
    )
    save_outputs(
        config,
        metrics,
        predictions,
        feature_scores,
        selected_features,
        models,
        best_model_name,
        X_test_selected,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    metrics = run(config)
    print("Experiment finished.")
    print(f"Artifacts: {config.output_dir}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
