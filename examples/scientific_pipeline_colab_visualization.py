from pathlib import Path
import sys
import warnings
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython import get_ipython
except Exception:
    get_ipython = lambda: None

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

print("\n========== PBL_RESEARCH SCIENTIFIC PIPELINE ==========\n")

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT = Path("/content/PBL_Research")

if not PROJECT.exists():
    PROJECT = Path.cwd()

sys.path.insert(0, str(PROJECT))

SATELLITE_DIR = Path("/content/data/raw")

if not SATELLITE_DIR.exists():
    SATELLITE_DIR = PROJECT / "data" / "raw"

ACTIVATE_DIR = PROJECT / "data" / "activate"

HSRL_DIR = PROJECT / "data" / "hsrl"

OUTPUT = PROJECT / "outputs"

OUTPUT.mkdir(
    exist_ok=True,
)

print("Project:", PROJECT)
print("Satellite:", SATELLITE_DIR)
print("ACTIVATE:", ACTIVATE_DIR)
print("HSRL:", HSRL_DIR)
print("Output:", OUTPUT)

# -----------------------------------------------------
# Imports
# -----------------------------------------------------

from PBL_Research.acquisition.hsrl import HSRLReader

from PBL_Research.collocation.spatial import SpatialCollocator
from PBL_Research.collocation.temporal import TemporalCollocator

from PBL_Research.builder.scientific_builder import ScientificBuilder

from PBL_Research.acquisition.activate import (
    ACTIVATE,
    ACTIVATEProduct,
)

# Download HSRL data if necessary

activate = ACTIVATE(
    ACTIVATEProduct.HSRL2,
)

if not HSRL_DIR.exists():

    HSRL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

hsrl_files = sorted(
    HSRL_DIR.rglob("*.h5")
)

if len(hsrl_files) == 0:

    print(
        "Downloading ACTIVATE HSRL-2 data..."
    )

    activate.download(
        temporal=(
            START_DATE,
            END_DATE,
        ),
        directory=HSRL_DIR,
    )

    hsrl_files = sorted(
        HSRL_DIR.rglob("*.h5")
    )

# -----------------------------------------------------
# Discover HSRL files
# -----------------------------------------------------

hsrl_files = sorted(
    HSRL_DIR.rglob("*.h5")
)

if len(hsrl_files) == 0:

    raise RuntimeError(
        "No HSRL files found."
    )

print(
    f"\nHSRL files: {len(hsrl_files)}"
)

# -----------------------------------------------------
# Discover satellite rasters
# -----------------------------------------------------

satellite_files = sorted(
    SATELLITE_DIR.rglob("*.tif")
)

if len(satellite_files) == 0:

    raise RuntimeError(
        "No satellite rasters found."
    )

print(
    f"Satellite rasters: {len(satellite_files)}"
)

# -----------------------------------------------------
# Read HSRL observations
# -----------------------------------------------------

print("\nLoading HSRL observations...")

hsrl_datasets = []

for file in hsrl_files:

    try:

        reader = HSRLReader(file)

        dataset = reader.read_dataset()

        available = reader.available_variables()

        print(
            file.name,
            "->",
            available,
        )

        hsrl_datasets.append(
            dataset,
        )

    except Exception as exc:

        print(
            file.name,
            exc,
        )

if len(hsrl_datasets) == 0:

    raise RuntimeError(
        "No HSRL dataset could be loaded."
    )

print(
    f"\nLoaded {len(hsrl_datasets)} HSRL datasets."
)

# -----------------------------------------------------
# Target variable
# -----------------------------------------------------

TARGET_VARIABLE = "mixed_layer_height"

print(
    "Scientific target:",
    TARGET_VARIABLE,
)

builder = ScientificBuilder()

spatial = SpatialCollocator()

temporal = TemporalCollocator()

# -----------------------------------------------------
# Read satellite features
# -----------------------------------------------------

print("\nLoading satellite features...")

satellite_frames = []

for raster in satellite_files:

    try:

        import rasterio

        with rasterio.open(raster) as src:

            image = src.read().astype(np.float32)

            if src.nodata is not None:

                image[image == src.nodata] = np.nan

            row = {

                "file": raster.name,

                "latitude": (
                    src.bounds.bottom
                    + src.bounds.top
                ) / 2.0,

                "longitude": (
                    src.bounds.left
                    + src.bounds.right
                ) / 2.0,

                "timestamp": pd.Timestamp(
                    raster.stat().st_mtime,
                    unit="s",
                ),

            }

            for band in range(image.shape[0]):

                values = image[band]

                row[f"band_{band+1}_mean"] = np.nanmean(values)
                row[f"band_{band+1}_std"] = np.nanstd(values)
                row[f"band_{band+1}_min"] = np.nanmin(values)
                row[f"band_{band+1}_max"] = np.nanmax(values)

            satellite_frames.append(row)

    except Exception as exc:

        print(raster.name, exc)

satellite_df = pd.DataFrame(
    satellite_frames,
)

print(
    satellite_df.shape,
)

# -----------------------------------------------------
# Scientific collocation
# -----------------------------------------------------

print("\nSpatial collocation...")

for hsrl in hsrl_datasets:

    spatial_df = spatial.collocate(

        satellite_df,

        hsrl,

    )

    print(

        "Spatial matches:",

        len(spatial_df),

    )

    temporal_df = temporal.collocate(

        spatial_df,

        hsrl,

    )

    print(

        "Temporal matches:",

        len(temporal_df),

    )

    builder.add(

        temporal_df,

    )

# -----------------------------------------------------
# Build scientific dataset
# -----------------------------------------------------

dataset = builder.build()

print(

    "\nScientific dataset:",

    dataset.shape,

)

if TARGET_VARIABLE not in dataset.columns:

    raise RuntimeError(

        f"{TARGET_VARIABLE} not found."

    )

dataset = dataset.dropna(

    subset=[

        TARGET_VARIABLE,

    ]

)

print(

    "After filtering:",

    dataset.shape,

)

X = dataset.drop(

    columns=[

        TARGET_VARIABLE,

    ]

)

X = X.select_dtypes(

    include=np.number,

).fillna(0)

y = dataset[

    TARGET_VARIABLE

].astype(

    np.float32,

)

print(

    "\nFeatures:",

    X.shape,

)

print(

    "Target:",

    y.shape,

)

# -----------------------------------------------------
# Train / validation / test split
# -----------------------------------------------------

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

X_train = scaler.fit_transform(

    X_train,

)

X_validation = scaler.transform(

    X_validation,

)

X_test = scaler.transform(

    X_test,

)

print(

    "\nTrain:",

    len(X_train),

)

print(

    "Validation:",

    len(X_validation),

)

print(

    "Test:",

    len(X_test),

)

# -----------------------------------------------------
# Model zoo
# -----------------------------------------------------

print("\nDefining candidate models...")

models = {

    "LinearRegression": LinearRegression(),

    "Ridge": Ridge(
        alpha=1.0,
    ),

    "Lasso": Lasso(
        alpha=0.001,
    ),

    "ElasticNet": ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
    ),

    "DecisionTree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42,
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42,
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
    ),

    "SVR": SVR(
        C=10.0,
        epsilon=0.1,
    ),

}

# -----------------------------------------------------
# Training
# -----------------------------------------------------

print("\nTraining candidate models...")

trained_models = {}

validation_scores = {}

for name, model in models.items():

    print(f"\n{name}")

    model.fit(

        X_train,

        y_train,

    )

    score = model.score(

        X_validation,

        y_validation,

    )

    trained_models[name] = model

    validation_scores[name] = float(score)

    print(

        f"Validation R²: {score:.4f}"

    )

# -----------------------------------------------------
# Best model
# -----------------------------------------------------

best_name = max(

    validation_scores,

    key=validation_scores.get,

)

best_model = trained_models[

    best_name

]

print("\nBest model:", best_name)

print(

    "Validation R²:",

    validation_scores[best_name],

)

# -----------------------------------------------------
# Test evaluation
# -----------------------------------------------------

print("\nEvaluating best model...")

predictions = best_model.predict(

    X_test,

)

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score,

)

mae = mean_absolute_error(

    y_test,

    predictions,

)

mse = mean_squared_error(

    y_test,

    predictions,

)

rmse = np.sqrt(

    mse,

)

r2 = r2_score(

    y_test,

    predictions,

)

print("\n========== FINAL RESULTS ==========")

print(

    f"MAE  : {mae:.4f}"

)

print(

    f"MSE  : {mse:.4f}"

)

print(

    f"RMSE : {rmse:.4f}"

)

print(

    f"R²   : {r2:.4f}"

)

# -----------------------------------------------------
# Save artifacts
# -----------------------------------------------------

joblib.dump(

    best_model,

    OUTPUT / "scientific_model.joblib",

)

joblib.dump(

    scaler,

    OUTPUT / "scientific_scaler.joblib",

)

results = {

    "best_model": best_name,

    "validation_scores": validation_scores,

    "test": {

        "MAE": float(mae),

        "MSE": float(mse),

        "RMSE": float(rmse),

        "R2": float(r2),

    },

}

with open(

    OUTPUT / "scientific_results.json",

    "w",

) as f:

    json.dump(

        results,

        f,

        indent=4,

    )

print("\nScientific model saved.")

print("Results saved.")

# -----------------------------------------------------
# Colab visualization
# -----------------------------------------------------

print("\nDisplaying scientific figures...")

plt.ioff()

# 1. Observed vs predicted
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, predictions, alpha=0.6)
lo = float(min(np.min(y_test), np.min(predictions)))
hi = float(max(np.max(y_test), np.max(predictions)))
ax.plot([lo, hi], [lo, hi], linestyle="--")
ax.set_xlabel("Observed mixed layer height")
ax.set_ylabel("Predicted mixed layer height")
ax.set_title(f"Observed vs Predicted — {best_name}")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / "observed_vs_predicted.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

# 2. Residual distribution
residuals = np.asarray(y_test) - np.asarray(predictions)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=40)
ax.axvline(0.0, linestyle="--")
ax.set_xlabel("Residual (observed - predicted)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / "residuals.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

# 3. Validation model comparison
ordered_scores = sorted(validation_scores.items(), key=lambda item: item[1], reverse=True)
names = [name for name, _ in ordered_scores]
scores = [score for _, score in ordered_scores]
fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(names))))
ax.barh(names[::-1], scores[::-1])
ax.set_xlabel("Validation R²")
ax.set_title("Validation Performance by Model")
ax.grid(True, axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / "validation_model_comparison.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

# 4. Feature importance / coefficients
feature_names = list(X.columns)
importance = None
if hasattr(best_model, "feature_importances_"):
    importance = np.asarray(best_model.feature_importances_, dtype=float)
elif hasattr(best_model, "coef_"):
    coef = np.asarray(best_model.coef_, dtype=float)
    importance = np.abs(coef).ravel()

if importance is not None and len(importance) == len(feature_names):
    top_k = min(20, len(feature_names))
    order = np.argsort(importance)[-top_k:]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.38 * top_k)))
    ax.barh(np.asarray(feature_names)[order], importance[order])
    ax.set_xlabel("Importance" if hasattr(best_model, "feature_importances_") else "Absolute coefficient")
    ax.set_title(f"Top {top_k} Feature Contributions — {best_name}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
else:
    print(f"Feature-importance plot not available for {best_name}.")

# 5. PBLH / target time series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.asarray(y_test))
ax.set_xlabel("Test sample index")
ax.set_ylabel("Observed mixed layer height")
ax.set_title("PBLH Test-Series View")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT / "pbl_height_timeseries.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Figures saved to:", OUTPUT)
for path in sorted(OUTPUT.glob("*.png")):
    print(" -", path.name)

print("\n========== PIPELINE FINISHED ==========")
