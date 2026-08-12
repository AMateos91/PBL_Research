from PBL_Research.export import FigureExporter
from pathlib import Path
import sys
import warnings
import json

import numpy as np
import pandas as pd
import rasterio
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

print("\n========== PBL_RESEARCH FULL DEMO PIPELINE ==========\n")

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT = Path("/content/PBL_Research")

if not PROJECT.exists():
    PROJECT = Path.cwd()

sys.path.insert(0, str(PROJECT))

RAW = Path("/content/data/raw")

if not RAW.exists():
    RAW = PROJECT / "data" / "raw"

OUTPUT = PROJECT / "outputs"
OUTPUT.mkdir(exist_ok=True)

print("Project:", PROJECT)
print("Raw data:", RAW)
print("Output:", OUTPUT)

# -----------------------------------------------------
# Data discovery
# -----------------------------------------------------

rasters = sorted(RAW.rglob("*.tif"))

print(f"\nRaster files found: {len(rasters)}")

if len(rasters) == 0:
    raise RuntimeError("No TIFF files found.")

# -----------------------------------------------------
# Raster loading
# -----------------------------------------------------

print("\n[1] Loading raster samples")

samples = []

for raster in rasters:

    try:

        with rasterio.open(raster) as src:

            array = src.read().astype(np.float32)

            nodata = src.nodata

            if nodata is not None:
                array[array == nodata] = np.nan

            samples.append(
                {
                    "path": raster,
                    "name": raster.stem,
                    "data": array,
                    "profile": src.profile,
                    "height": src.height,
                    "width": src.width,
                    "bands": src.count,
                    "crs": str(src.crs),
                    "transform": src.transform,
                }
            )

            print(
                f"{raster.name:40s}{array.shape}"
            )

    except Exception as exc:

        print(f"Skipping {raster.name}: {exc}")

if len(samples) == 0:
    raise RuntimeError("No valid raster could be loaded.")

# -----------------------------------------------------
# Feature extraction
# -----------------------------------------------------

print("\n[2] Creating feature matrix")

records = []

for sample in samples:

    img = sample["data"]

    row = {
        "file": sample["name"],
        "bands": sample["bands"],
        "height": sample["height"],
        "width": sample["width"],
    }

    for b in range(img.shape[0]):

        band = img[b]

        row[f"band_{b+1}_mean"] = np.nanmean(band)
        row[f"band_{b+1}_std"] = np.nanstd(band)
        row[f"band_{b+1}_min"] = np.nanmin(band)
        row[f"band_{b+1}_max"] = np.nanmax(band)
        row[f"band_{b+1}_median"] = np.nanmedian(band)
        row[f"band_{b+1}_q25"] = np.nanpercentile(band, 25)
        row[f"band_{b+1}_q75"] = np.nanpercentile(band, 75)

    records.append(row)

dataset = pd.DataFrame(records)

print(dataset.head())

numeric_columns = dataset.select_dtypes(
    include=np.number
).columns.tolist()

X = dataset[numeric_columns].fillna(0.0)

print("\nFeature matrix:", X.shape)

# -----------------------------------------------------
# Synthetic target
# -----------------------------------------------------

print("\n[3] Creating synthetic regression target")

rng = np.random.default_rng(42)

y = (
    X.mean(axis=1)
    + rng.normal(0, 0.15, len(X))
)

# -----------------------------------------------------
# Train / validation / test split
# -----------------------------------------------------

print("\n[4] Splitting dataset")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_validation)}")
print(f"Test: {len(X_test)}")

# -----------------------------------------------------
# Model zoo
# -----------------------------------------------------

print("\n[5] Defining candidate models")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=42,
    ),
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "SVR": SVR(C=10.0, epsilon=0.1),
}

# -----------------------------------------------------
# Training
# -----------------------------------------------------

print("\n[6] Training models")

trained_models = {}
validation_scores = {}

for name, model in models.items():

    print(f"Training {name} ...")

    model.fit(X_train, y_train)

    score = model.score(X_validation, y_validation)

    trained_models[name] = model
    validation_scores[name] = score

    print(f"Validation R2: {score:.4f}\n")

best_name = max(
    validation_scores,
    key=validation_scores.get,
)

best_model = trained_models[best_name]

print("Best validation model:", best_name)
print(
    f"Best validation R2: {validation_scores[best_name]:.4f}"
)

joblib.dump(
    best_model,
    OUTPUT / "best_model.joblib",
)

joblib.dump(
    scaler,
    OUTPUT / "scaler.joblib",
)

with open(
    OUTPUT / "validation_scores.json",
    "w",
) as f:

    json.dump(
        validation_scores,
        f,
        indent=2,
    )

print("\nTraining stage completed.")

# -----------------------------------------------------
# Test evaluation
# -----------------------------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

print("\n[7] Evaluating models on test set")

results = []

predictions = {}

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred,
    )

    bias = np.mean(
        y_pred - y_test
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "BIAS": bias,
        }
    )

    predictions[name] = y_pred

results = pd.DataFrame(results)

results = results.sort_values(
    by="R2",
    ascending=False,
).reset_index(drop=True)

print("\nTest results")
print(results)

results.to_csv(
    OUTPUT / "model_comparison.csv",
    index=False,
)

# -----------------------------------------------------
# Best model
# -----------------------------------------------------

best_name = results.iloc[0]["Model"]

best_model = trained_models[
    best_name
]

print("\nBest model on TEST")

print(best_name)

# -----------------------------------------------------
# Predictions
# -----------------------------------------------------

best_prediction = best_model.predict(
    X_test,
)

prediction_table = pd.DataFrame(
    {
        "Observed": y_test,
        "Predicted": best_prediction,
        "Residual": best_prediction - y_test,
        "AbsoluteError": np.abs(
            best_prediction - y_test
        ),
    }
)

prediction_table.to_csv(
    OUTPUT / "predictions.csv",
    index=False,
)

print(
    "\nPredictions saved."
)

# -----------------------------------------------------
# Save metrics
# -----------------------------------------------------

summary = {
    "best_model": best_name,
    "R2": float(
        results.iloc[0]["R2"]
    ),
    "RMSE": float(
        results.iloc[0]["RMSE"]
    ),
    "MAE": float(
        results.iloc[0]["MAE"]
    ),
    "MSE": float(
        results.iloc[0]["MSE"]
    ),
    "BIAS": float(
        results.iloc[0]["BIAS"]
    ),
}

with open(
    OUTPUT / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )

print("\nSummary")

for k, v in summary.items():

    print(
        f"{k}: {v}"
    )

# -----------------------------------------------------
# Residual statistics
# -----------------------------------------------------

residuals = (
    best_prediction
    - y_test
)

print("\nResidual statistics")

print(
    f"Mean residual : {np.mean(residuals):.6f}"
)

print(
    f"Std residual  : {np.std(residuals):.6f}"
)

print(
    f"Median residual : {np.median(residuals):.6f}"
)

print(
    "\nEvaluation stage completed."
)

# -----------------------------------------------------
# Visualization
# -----------------------------------------------------

import matplotlib.pyplot as plt

print("\n[8] Generating figures")

# -----------------------------------------------------
figure_exporter = FigureExporter()

# Model comparison
# -----------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["R2"],
)

plt.xticks(rotation=45, ha="right")

plt.ylabel("R²")

plt.title("Model comparison")

plt.tight_layout()

plt.savefig(
    OUTPUT / "model_comparison.png",
    dpi=300,
)

plt.close()

# -----------------------------------------------------
# Prediction vs observation
# -----------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    y_test,
    best_prediction,
)

minimum = min(
    np.min(y_test),
    np.min(best_prediction),
)

maximum = max(
    np.max(y_test),
    np.max(best_prediction),
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
)

ax.set_xlabel("Observed")
ax.set_ylabel("Predicted")
ax.set_title(best_name)

fig.tight_layout()

figure_exporter.export(
    fig,
    str(OUTPUT / "prediction_vs_observation.png"),
)

plt.close(fig)

# -----------------------------------------------------
# Residual histogram
# -----------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    residuals,
    bins=20,
)

ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.set_title("Residual distribution")

fig.tight_layout()

figure_exporter.export(
    fig,
    str(OUTPUT / "residual_distribution.png"),
)

plt.close(fig)

# -----------------------------------------------------
# Residual scatter
# -----------------------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    best_prediction,
    residuals,
)

plt.axhline(
    0,
    linestyle="--",
)

plt.xlabel("Prediction")

plt.ylabel("Residual")

plt.title("Residual analysis")

plt.tight_layout()

plt.savefig(
    OUTPUT / "residual_scatter.png",
    dpi=300,
)

plt.close()

# -----------------------------------------------------
# Feature importance
# -----------------------------------------------------

if hasattr(
    best_model,
    "feature_importances_",
):

    importance = pd.DataFrame(
        {
            "Feature": numeric_columns,
            "Importance": best_model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT / "feature_importance.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        importance["Feature"][:20],
        importance["Importance"][:20],
    )

    ax.invert_yaxis()

    fig.tight_layout()

    figure_exporter.export(
        fig,
        str(OUTPUT / "feature_importance.png"),
    )

    plt.close(fig)

# -----------------------------------------------------
# Ranking
# -----------------------------------------------------

print("\nModel ranking\n")

print(results)

# -----------------------------------------------------
# Export complete report
# -----------------------------------------------------

results.to_json(
    OUTPUT / "results.json",
    orient="records",
    indent=4,
)

prediction_table.to_json(
    OUTPUT / "predictions.json",
    orient="records",
    indent=4,
)

print("\nFiles generated:\n")

for file in sorted(
    OUTPUT.iterdir()
):

    print(file.name)

print("\nPipeline completed successfully.")

print("\nBest model:", best_name)

print(
    f"R²   : {summary['R2']:.4f}"
)

print(
    f"RMSE : {summary['RMSE']:.4f}"
)

print(
    f"MAE  : {summary['MAE']:.4f}"
)

print(
    f"BIAS : {summary['BIAS']:.4f}"
)

print("\n========== END ==========\n")
