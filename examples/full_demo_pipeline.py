from pathlib import Path
import sys
import importlib
import inspect
import numpy as np
import rasterio


print("\n========== PBL_RESEARCH FULL DEMO PIPELINE ==========\n")


# -----------------------------------------------------
# Paths
# -----------------------------------------------------

PROJECT = Path("/content/PBL_Research")

if not PROJECT.exists():
    PROJECT = Path.cwd()


sys.path.insert(
    0,
    str(PROJECT)
)


RAW = Path("/content/data/raw")

if not RAW.exists():
    RAW = PROJECT / "data" / "raw"


print("Project:", PROJECT)
print("Raw data:", RAW)


# -----------------------------------------------------
# Data discovery
# -----------------------------------------------------

rasters = list(
    RAW.rglob("*.tif")
)


print(
    f"\nRaster files found: {len(rasters)}"
)


if not rasters:
    raise RuntimeError(
        "No TIFF files found"
    )


# -----------------------------------------------------
# Raster loading
# -----------------------------------------------------

print("\n[1] Loading raster samples")


samples = []


for raster in rasters:

    with rasterio.open(raster) as src:

        array = src.read()

        samples.append(
            {
                "path": raster,
                "data": array,
                "profile": src.profile
            }
        )

        print(
            raster.name,
            array.shape
        )


# -----------------------------------------------------
# Basic feature extraction
# -----------------------------------------------------

print("\n[2] Creating feature matrix")


X = []


for sample in samples:

    data = sample["data"].astype(
        np.float32
    )


    features = [

        np.nanmean(data),

        np.nanstd(data),

        np.nanmin(data),

        np.nanmax(data)

    ]


    X.append(features)



X = np.asarray(X)


print(
    "Feature matrix:",
    X.shape
)



# -----------------------------------------------------
# Discover ML modules
# -----------------------------------------------------

print("\n[3] Searching ML package")


ml_path = PROJECT / "ml"


if ml_path.exists():

    print(
        "ML package detected"
    )

    for p in ml_path.rglob("*.py"):

        print(
            " ",
            p.relative_to(PROJECT)
        )

else:

    print(
        "No ML directory found"
    )



# -----------------------------------------------------
# Baseline training
# -----------------------------------------------------

print("\n[4] Training baseline model")


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Placeholder target generated from raster statistics.
# Replace with ACTIVATE/MERRA target once connected.

y = np.mean(
    X,
    axis=1
)



if len(X) > 2:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )


    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    score = model.score(
        X_test,
        y_test
    )


    print(
        "R2:",
        score
    )


else:

    print(
        "Not enough samples for split"
    )



# -----------------------------------------------------
# Save outputs
# -----------------------------------------------------

out = PROJECT / "outputs"

out.mkdir(
    exist_ok=True
)


np.save(
    out / "features.npy",
    X
)


print(
    "\nFeatures saved:",
    out / "features.npy"
)


print(
    "\n========== PIPELINE FINISHED =========="
)
