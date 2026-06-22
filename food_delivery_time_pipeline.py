"""
Food Delivery Time Prediction - Clean Commented Script

This script is a readable Python version of the notebook project.
It:
1. loads the dataset
2. cleans and preprocesses the data
3. trains Linear Regression and Decision Tree models
4. compares their performance
5. saves the trained model and report files
"""

import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def normalize_name(name: str) -> str:
    """Convert a column name into a normalized lowercase form."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the delivery time target column from likely names."""
    columns = list(df.columns)
    normalized = {col: normalize_name(col) for col in columns}

    exact_priority = {
        "delivery_time",
        "delivery_time_min",
        "delivery_time_minutes",
        "time",
        "time_taken",
        "time_taken_min",
        "time_taken_minutes",
    }

    exact_matches = [col for col in columns if normalized[col] in exact_priority]
    if exact_matches:
        return exact_matches[0]

    delivery_time_candidates = [
        col for col in columns if "delivery" in normalized[col] and "time" in normalized[col]
    ]
    if delivery_time_candidates:
        return delivery_time_candidates[0]

    time_candidates = [col for col in columns if "time" in normalized[col]]
    if time_candidates:
        numeric_candidates = [
            col for col in time_candidates if str(df[col].dtype).startswith(("int", "float"))
        ]
        return numeric_candidates[0] if numeric_candidates else time_candidates[0]

    raise ValueError("No delivery time target column found.")


def make_target_numeric(series: pd.Series) -> pd.Series:
    """Convert the target column to numeric safely."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().mean() >= 0.6:
        return converted

    return (
        series.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def detect_identifier_columns(columns) -> list:
    """Find columns that look like identifiers and should not be used as predictors."""
    identifier_columns = []
    for col in columns:
        normalized = normalize_name(col)
        if normalized == "id" or normalized.endswith("_id") or normalized in {"order_id", "delivery_id"}:
            identifier_columns.append(col)
    return identifier_columns


def build_preprocessor(numeric_cols, categorical_cols):
    """Create a preprocessing pipeline for numeric and categorical features."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def safe_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error while ignoring zero targets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def main():
    csv_path = Path("finaldata.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Please place finaldata.csv in the same folder as this script.")

    # Load dataset.
    df = pd.read_csv(csv_path)
    print("Loaded dataset with shape:", df.shape)

    # Basic cleaning.
    df = df.dropna(how="all").copy()
    df = df.dropna(axis=1, how="all")

    # Detect target and convert it to numeric values.
    target_col = detect_target_column(df)
    df[target_col] = make_target_numeric(df[target_col])
    df = df.dropna(subset=[target_col]).copy()

    # Separate input features and target.
    X_full = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Remove identifier-like columns such as Order_ID.
    identifier_columns = detect_identifier_columns(X_full.columns)
    model_columns = [col for col in X_full.columns if col not in identifier_columns]
    X = X_full[model_columns].copy()

    # Split columns into numeric and categorical groups.
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    full_numeric_features = X_full.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    full_categorical_features = [c for c in X_full.columns if c not in full_numeric_features]

    # Fill missing values in the original full feature table.
    if full_numeric_features:
        X_full[full_numeric_features] = X_full[full_numeric_features].fillna(X_full[full_numeric_features].median())
    if full_categorical_features:
        for col in full_categorical_features:
            mode = X_full[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            X_full[col] = X_full[col].fillna(fill_value)

    # Keep the final model feature table after removing identifiers.
    X = X_full[model_columns].copy()

    # Save cleaned data for submission.
    cleaned_df = X_full.copy()
    cleaned_df[target_col] = y
    cleaned_df.to_csv("cleaned_data.csv", index=False)

    # Use the same split indices for both the with-ID and without-ID experiments.
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, random_state=42)
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()
    X_full_train = X_full.loc[train_idx].copy()
    X_full_test = X_full.loc[test_idx].copy()

    # Build preprocessors and pipelines.
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor_with_id = build_preprocessor(full_numeric_features, full_categorical_features)

    linear_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    tree_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42)),
    ])

    linear_model_with_id = Pipeline([
        ("preprocessor", preprocessor_with_id),
        ("model", LinearRegression()),
    ])

    # Train the models.
    linear_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    linear_model_with_id.fit(X_full_train, y_train)

    # Make predictions.
    y_pred_linear = linear_model.predict(X_test)
    y_pred_tree = tree_model.predict(X_test)
    y_pred_linear_with_id = linear_model_with_id.predict(X_full_test)

    # Build the main comparison table.
    metrics_df = pd.DataFrame(
        {
            "MAE": [
                mean_absolute_error(y_test, y_pred_linear),
                mean_absolute_error(y_test, y_pred_tree),
            ],
            "RMSE": [
                mean_squared_error(y_test, y_pred_linear) ** 0.5,
                mean_squared_error(y_test, y_pred_tree) ** 0.5,
            ],
            "MAPE (%)": [
                safe_mape(y_test, y_pred_linear),
                safe_mape(y_test, y_pred_tree),
            ],
            "R2 Score": [
                r2_score(y_test, y_pred_linear),
                r2_score(y_test, y_pred_tree),
            ],
        },
        index=["Linear Regression", "Decision Tree"],
    )

    # Compare Linear Regression with and without Order_ID.
    id_experiment_df = pd.DataFrame(
        {
            "MAE": [
                mean_absolute_error(y_test, y_pred_linear_with_id),
                mean_absolute_error(y_test, y_pred_linear),
            ],
            "RMSE": [
                mean_squared_error(y_test, y_pred_linear_with_id) ** 0.5,
                mean_squared_error(y_test, y_pred_linear) ** 0.5,
            ],
            "R2 Score": [
                r2_score(y_test, y_pred_linear_with_id),
                r2_score(y_test, y_pred_linear),
            ],
        },
        index=["Linear Regression + Order_ID", "Linear Regression without Order_ID"],
    )

    # Cross-validation provides a more stable estimate of performance.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = []
    for name, model in [("Linear Regression", linear_model), ("Decision Tree", tree_model)]:
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        cv_summary.append(
            {
                "Model": name,
                "CV MAE Mean": mae_scores.mean(),
                "CV MAE Std": mae_scores.std(),
                "CV R2 Mean": r2_scores.mean(),
                "CV R2 Std": r2_scores.std(),
            }
        )
    cv_df = pd.DataFrame(cv_summary).set_index("Model")

    # Save a clean dashboard figure for the report.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    residual_linear = y_test.values - y_pred_linear
    residual_tree = y_test.values - y_pred_tree
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))

    min_val = float(min(y_test.min(), np.min(y_pred_linear), np.min(y_pred_tree)))
    max_val = float(max(y_test.max(), np.max(y_pred_linear), np.max(y_pred_tree)))
    ideal_line = [min_val, max_val]

    axes[0, 0].scatter(y_test, y_pred_linear, alpha=0.7, s=35, color="#1f77b4")
    axes[0, 0].plot(ideal_line, ideal_line, "r--", linewidth=1.5)
    axes[0, 0].set_title("Linear Regression: Actual vs Predicted")
    axes[0, 0].set_xlabel("Actual Delivery Time")
    axes[0, 0].set_ylabel("Predicted Delivery Time")

    axes[0, 1].scatter(y_test, y_pred_tree, alpha=0.7, s=35, color="#ff7f0e")
    axes[0, 1].plot(ideal_line, ideal_line, "r--", linewidth=1.5)
    axes[0, 1].set_title("Decision Tree: Actual vs Predicted")
    axes[0, 1].set_xlabel("Actual Delivery Time")
    axes[0, 1].set_ylabel("Predicted Delivery Time")

    error_metrics = metrics_df[["MAE", "RMSE", "MAPE (%)"]]
    x = np.arange(len(error_metrics.columns))
    width = 0.35
    axes[1, 0].bar(x - width / 2, error_metrics.loc["Linear Regression"], width, label="Linear Regression", color="#1f77b4")
    axes[1, 0].bar(x + width / 2, error_metrics.loc["Decision Tree"], width, label="Decision Tree", color="#ff7f0e")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(error_metrics.columns, rotation=15)
    axes[1, 0].set_title("Error Metric Comparison")
    axes[1, 0].set_ylabel("Metric Value")
    axes[1, 0].legend()

    axes[1, 1].bar(metrics_df.index, metrics_df["R2 Score"], color=["#1f77b4", "#ff7f0e"])
    axes[1, 1].set_title("R2 Score Comparison")
    axes[1, 1].set_ylabel("R2 Score")
    axes[1, 1].set_ylim(0, 1)

    heatmap_values = metrics_df.round(2).values
    heatmap = axes[2, 0].imshow(heatmap_values, cmap="YlGnBu", aspect="auto")
    axes[2, 0].set_title("Performance Metric Heatmap")
    axes[2, 0].set_xticks(np.arange(len(metrics_df.columns)))
    axes[2, 0].set_xticklabels(metrics_df.columns, rotation=25, ha="right")
    axes[2, 0].set_yticks(np.arange(len(metrics_df.index)))
    axes[2, 0].set_yticklabels(metrics_df.index)
    for row in range(heatmap_values.shape[0]):
        for col in range(heatmap_values.shape[1]):
            axes[2, 0].text(col, row, heatmap_values[row, col], ha="center", va="center", color="black")
    fig.colorbar(heatmap, ax=axes[2, 0], fraction=0.046, pad=0.04)

    axes[2, 1].boxplot(
        [np.abs(residual_linear), np.abs(residual_tree)],
        labels=["Linear Regression", "Decision Tree"],
        patch_artist=True,
        boxprops=dict(facecolor="#d9eaf7"),
        medianprops=dict(color="#d62728"),
    )
    axes[2, 1].set_title("Absolute Residual Distribution")
    axes[2, 1].set_ylabel("Absolute Error")

    plt.suptitle("Linear Regression vs Decision Tree Comparison Dashboard", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig("model_comparison_dashboard.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Save the trained model.
    with open("model.pkl", "wb") as f:
        pickle.dump(linear_model, f)

    # Save useful metadata for the report and future reuse.
    model_info = {
        "target_column": target_col,
        "feature_columns": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "removed_identifier_columns": identifier_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {
            "linear_regression": {metric_key.lower().replace(" (%)", "_percent").replace(" ", "_"): float(value)
                                  for metric_key, value in metrics_df.loc["Linear Regression"].items()},
            "decision_tree_shallow": {metric_key.lower().replace(" (%)", "_percent").replace(" ", "_"): float(value)
                                      for metric_key, value in metrics_df.loc["Decision Tree"].items()},
        },
        "cross_validation": {
            row_name: {metric: float(value) for metric, value in row.items()}
            for row_name, row in cv_df.to_dict(orient="index").items()
        },
        "order_id_experiment": {
            row_name: {metric: float(value) for metric, value in row.items()}
            for row_name, row in id_experiment_df.to_dict(orient="index").items()
        },
    }

    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("\nFinal Performance Summary")
    print(metrics_df.round(3))
    print("\nCross Validation Summary")
    print(cv_df.round(3))
    print("\nOrder_ID Experiment")
    print(id_experiment_df.round(3))


if __name__ == "__main__":
    main()

