from pathlib import Path
import sys
import warnings
import json

import joblib
import numpy as np
import pandas as pd

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

from acquisition.hsrl import HSRLReader

from collocation.spatial import SpatialCollocator
from collocation.temporal import TemporalCollocator

from builder.scientific_builder import ScientificBuilder

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

print("\n========== PIPELINE FINISHED ==========")
