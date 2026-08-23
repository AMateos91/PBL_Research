from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    display = None

warnings.filterwarnings("ignore")

PROJECT = Path("/content/PBL_Research")
if not PROJECT.exists():
    PROJECT = Path.cwd()

sys.path.insert(0, str(PROJECT))

RAW = Path("/content/data/raw")
if not RAW.exists():
    RAW = PROJECT / "data" / "raw"

OUTPUT = PROJECT / "outputs" / "scientific"
OUTPUT.mkdir(parents=True, exist_ok=True)

from PBL_Research.acquisition.hdf import HDFReader
from PBL_Research.builder.scientific_builder import ScientificBuilder
from PBL_Research.collocation.temporal import TemporalCollocator
from PBL_Research.datasets.scaling import Scaling
from PBL_Research.datasets.split import Split
from PBL_Research.evaluation import RegressionEvaluator
from PBL_Research.export import TableExporter, ModelExporter
from PBL_Research.features import FeaturePipeline
from PBL_Research.features.heterogeneity import Heterogeneity
from PBL_Research.ml.classical import (
    LinearRegressionModel,
    RidgeRegressionModel,
    RandomForestRegressionModel,
)
from PBL_Research.ml.explainability import PermutationImportance
from PBL_Research.ml.inference import Predictor
from PBL_Research.utils.constants import ExportFormat, ScalingMethod
from PBL_Research.visualization import Plotter, FeatureImportance, TimeSeriesVisualizer

R1_PATH = RAW / "ACTIVATE-HSRL2-Cloud_KingAir_20220111_R1_L1.h5"
R3_PATH = RAW / "ACTIVATE-HSRL2_KingAir_20220111_R3_L1.h5"

if not R1_PATH.exists():
    candidates = sorted(RAW.glob("*R1_L1.h5"))
    if len(candidates) == 1:
        R1_PATH = candidates[0]

if not R3_PATH.exists():
    candidates = sorted(RAW.glob("*R3_L1.h5"))
    if len(candidates) == 1:
        R3_PATH = candidates[0]

if not R1_PATH.exists() or not R3_PATH.exists():
    raise FileNotFoundError(
        f"R1={R1_PATH} exists={R1_PATH.exists()} | "
        f"R3={R3_PATH} exists={R3_PATH.exists()}"
    )

def resolve_dataset(reader: HDFReader, candidates: list[str], label: str) -> str:
    available = set(reader.datasets)
    for path in candidates:
        if path in available:
            print(f"{label}: {path}")
            return path
    raise KeyError(
        f"{label} not found. Tried: {candidates}. "
        f"Relevant datasets: {[p for p in reader.datasets if any(k.lower() in p.lower() for k in label.split())]}"
    )

def resolve_map(path: Path, candidates: dict[str, list[str]], label: str) -> dict[str, str]:
    with HDFReader(path) as reader:
        mapping = {}
        for name, options in candidates.items():
            mapping[name] = resolve_dataset(reader, options, f"{label}.{name}")
    return mapping

R1 = resolve_map(
    R1_PATH,
    {
        "time": ["/Nav_Data/gps_time", "/time"],
        "lat": ["/Nav_Data/gps_lat", "/lat"],
        "lon": ["/Nav_Data/gps_lon", "/lon"],
        "z": ["/DataProducts/Altitude", "/Altitude", "/z"],
        "temperature": ["/DataProducts/temperature_prfl", "/State/temperature_prfl", "/State/Temperature"],
        "cloud_height": ["/DataProducts/cloud_height"],
        "cloud_ext_prfl": ["/DataProducts/cloud_ext_prfl"],
    },
    "R1",
)

R3 = resolve_map(
    R3_PATH,
    {
        "time": ["/time", "/Nav_Data/gps_time"],
        "lat": ["/lat", "/Nav_Data/gps_lat"],
        "lon": ["/lon", "/Nav_Data/gps_lon"],
        "z": ["/DataProducts/Altitude", "/Altitude", "/z"],
        "backscatter": ["/DataProducts/532_bsc_cloud_screened"],
        "extinction": ["/DataProducts/532_ext", "/DataProducts/532_ext_prfl"],
        "temperature": ["/State/Temperature", "/State/temperature_prfl", "/DataProducts/temperature_prfl"],
        "pressure": ["/State/Pressure", "/State/pressure_prfl"],
    },
    "R3",
)

def read_array(path: Path, dataset_path: str) -> np.ndarray:
    with HDFReader(path) as reader:
        return np.asarray(reader.read(dataset_path))

def squeeze_1d(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after squeeze; shape={values.shape}")
    return values

def orient_profile(values, n_obs: int, n_levels: int, name: str) -> np.ndarray:
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; shape={values.shape}")
    if values.shape == (n_obs, n_levels):
        return values
    if values.shape == (n_levels, n_obs):
        return values.T
    raise ValueError(
        f"{name} shape={values.shape} is incompatible with "
        f"n_obs={n_obs}, n_levels={n_levels}"
    )

def time_values(values, unit_hint: str | None = None) -> pd.DatetimeIndex:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, utc=True)

    numeric = np.asarray(values, dtype=float).reshape(-1)
    if unit_hint == "hours_since_midnight":
        return (
            pd.Timestamp("2022-01-11", tz="UTC")
            + pd.to_timedelta(numeric, unit="h")
        )
    if unit_hint == "seconds_since_midnight":
        return (
            pd.Timestamp("2022-01-11", tz="UTC")
            + pd.to_timedelta(numeric, unit="s")
        )

    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return pd.to_datetime(numeric, unit="s", origin="unix", utc=True)

    scale = float(np.nanmedian(np.abs(finite)))

    if scale >= 1e14:
        return pd.to_datetime(numeric, unit="us", origin="unix", utc=True)
    if scale >= 1e11:
        return pd.to_datetime(numeric, unit="ms", origin="unix", utc=True)
    if scale >= 1e9:
        return pd.to_datetime(numeric, unit="s", origin="unix", utc=True)

    return (
        pd.Timestamp("2022-01-11", tz="UTC")
        + pd.to_timedelta(numeric, unit="s")
    )

def finite_profile_stat(profile, z, lower, upper, reducer):
    profile = np.asarray(profile, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = min(profile.size, z.size)
    profile = profile[:n]
    z = z[:n]
    mask = (
        np.isfinite(profile)
        & np.isfinite(z)
        & (z >= lower)
        & (z <= upper)
    )
    if mask.sum() == 0:
        return np.nan
    return float(reducer(profile[mask]))

def profile_gradient_min(profile, z, lower=0.0, upper=3000.0):
    profile = np.asarray(profile, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = min(profile.size, z.size)
    profile = profile[:n]
    z = z[:n]
    mask = (
        np.isfinite(profile)
        & (profile > 0)
        & np.isfinite(z)
        & (z >= lower)
        & (z <= upper)
    )
    if mask.sum() < 8:
        return np.nan
    zz = z[mask]
    yy = np.log10(profile[mask])
    order = np.argsort(zz)
    zz = zz[order]
    yy = yy[order]
    zz_unique, unique_idx = np.unique(zz, return_index=True)
    yy = yy[unique_idx]
    if zz_unique.size < 8:
        return np.nan
    gradient = np.gradient(yy, zz_unique)
    if not np.isfinite(gradient).any():
        return np.nan
    return float(np.nanmin(gradient))

def derive_pbl_height(profile, z, lower=100.0, upper=3000.0):
    profile = np.asarray(profile, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = min(profile.size, z.size)
    profile = profile[:n]
    z = z[:n]
    mask = (
        np.isfinite(profile)
        & (profile > 0)
        & np.isfinite(z)
        & (z >= lower)
        & (z <= upper)
    )
    if mask.sum() < 8:
        return np.nan
    zz = z[mask]
    yy = np.log10(profile[mask])
    order = np.argsort(zz)
    zz = zz[order]
    yy = yy[order]
    zz_unique, unique_idx = np.unique(zz, return_index=True)
    yy = yy[unique_idx]
    if zz_unique.size < 8:
        return np.nan
    gradient = np.gradient(yy, zz_unique)
    valid = np.isfinite(gradient)
    if not valid.any():
        return np.nan
    idx = np.nanargmin(np.where(valid, gradient, np.nan))
    return float(zz_unique[idx])

def profile_features(profile, z, prefix):
    result = {}
    for lower, upper, label in [
        (0.0, 500.0, "0_500"),
        (500.0, 1500.0, "500_1500"),
        (1500.0, 3000.0, "1500_3000"),
    ]:
        result[f"{prefix}_mean_{label}"] = finite_profile_stat(profile, z, lower, upper, np.nanmean)
        result[f"{prefix}_std_{label}"] = finite_profile_stat(profile, z, lower, upper, np.nanstd)
    result[f"{prefix}_mean_0_3000"] = finite_profile_stat(profile, z, 0.0, 3000.0, np.nanmean)
    result[f"{prefix}_std_0_3000"] = finite_profile_stat(profile, z, 0.0, 3000.0, np.nanstd)
    result[f"{prefix}_max_0_3000"] = finite_profile_stat(profile, z, 0.0, 3000.0, np.nanmax)
    result[f"{prefix}_gradient_min"] = profile_gradient_min(profile, z, 0.0, 3000.0)
    return result

def build_r3_frame():
    time = squeeze_1d(read_array(R3_PATH, R3["time"]), "R3.time")
    lat = squeeze_1d(read_array(R3_PATH, R3["lat"]), "R3.lat").astype(float)
    lon = squeeze_1d(read_array(R3_PATH, R3["lon"]), "R3.lon").astype(float)
    z = squeeze_1d(read_array(R3_PATH, R3["z"]), "R3.z").astype(float)

    backscatter = read_array(R3_PATH, R3["backscatter"]).astype(float)
    extinction = read_array(R3_PATH, R3["extinction"]).astype(float)
    temperature = read_array(R3_PATH, R3["temperature"]).astype(float)
    pressure = read_array(R3_PATH, R3["pressure"]).astype(float)

    if backscatter.ndim != 2:
        raise ValueError(f"Unexpected R3 backscatter shape: {backscatter.shape}")

    n_obs = backscatter.shape[0]
    n_levels = z.size

    if len(time) != n_obs or len(lat) != n_obs or len(lon) != n_obs:
        raise ValueError("R3 navigation/time dimensions do not match backscatter observations.")

    backscatter = orient_profile(backscatter, n_obs, n_levels, "R3.backscatter")
    extinction = orient_profile(extinction, n_obs, n_levels, "R3.extinction")
    temperature = orient_profile(temperature, n_obs, n_levels, "R3.temperature")
    pressure = orient_profile(pressure, n_obs, n_levels, "R3.pressure")

    rows = []
    times = time_values(time, unit_hint="seconds_since_midnight")

    for i in range(n_obs):
        row = {
            "time": times[i],
            "latitude": lat[i],
            "longitude": lon[i],
            "pbl_height_proxy_m": derive_pbl_height(backscatter[i], z),
            "backscatter_gradient_min": profile_gradient_min(backscatter[i], z),
        }
        row.update(profile_features(backscatter[i], z, "backscatter"))
        row.update(profile_features(extinction[i], z, "extinction"))
        row["temperature_mean_0_3000"] = finite_profile_stat(temperature[i], z, 0.0, 3000.0, np.nanmean)
        row["temperature_std_0_3000"] = finite_profile_stat(temperature[i], z, 0.0, 3000.0, np.nanstd)
        row["temperature_gradient_min"] = profile_gradient_min(temperature[i], z, 0.0, 3000.0)
        row["pressure_mean_0_3000"] = finite_profile_stat(pressure[i], z, 0.0, 3000.0, np.nanmean)
        row["pressure_std_0_3000"] = finite_profile_stat(pressure[i], z, 0.0, 3000.0, np.nanstd)
        row["pressure_gradient_min"] = profile_gradient_min(pressure[i], z, 0.0, 3000.0)
        rows.append(row)

    return pd.DataFrame(rows)

def build_r1_frame():
    time = squeeze_1d(read_array(R1_PATH, R1["time"]), "R1.time")
    lat = squeeze_1d(read_array(R1_PATH, R1["lat"]), "R1.lat").astype(float)
    lon = squeeze_1d(read_array(R1_PATH, R1["lon"]), "R1.lon").astype(float)
    z = squeeze_1d(read_array(R1_PATH, R1["z"]), "R1.z").astype(float)

    temperature = read_array(R1_PATH, R1["temperature"]).astype(float)
    cloud_height = squeeze_1d(read_array(R1_PATH, R1["cloud_height"]), "R1.cloud_height").astype(float)
    cloud_ext = read_array(R1_PATH, R1["cloud_ext_prfl"]).astype(float)

    temperature = orient_profile(temperature, len(time), z.size, "R1.temperature")
    cloud_ext = orient_profile(cloud_ext, len(time), z.size, "R1.cloud_ext_prfl")

    n_obs = temperature.shape[0]

    if len(time) != n_obs or len(lat) != n_obs or len(lon) != n_obs or len(cloud_height) != n_obs:
        raise ValueError("R1 navigation/product dimensions do not match temperature observations.")

    times = time_values(time, unit_hint="hours_since_midnight")
    rows = []

    for i in range(n_obs):
        row = {
            "time": times[i],
            "r1_latitude": lat[i],
            "r1_longitude": lon[i],
            "r1_cloud_height_m": cloud_height[i],
            "r1_temperature_mean_0_3000": finite_profile_stat(temperature[i], z, 0.0, 3000.0, np.nanmean),
            "r1_temperature_std_0_3000": finite_profile_stat(temperature[i], z, 0.0, 3000.0, np.nanstd),
            "r1_temperature_gradient_min": profile_gradient_min(temperature[i], z, 0.0, 3000.0),
            "r1_cloud_ext_mean_0_3000": finite_profile_stat(cloud_ext[i], z, 0.0, 3000.0, np.nanmean),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def heterogeneity_summary(frame):
    data = xr.Dataset(
        {
            "extinction": ("observation", frame["extinction_mean_0_3000"].to_numpy()),
            "temperature": ("observation", frame["temperature_mean_0_3000"].to_numpy()),
            "pressure": ("observation", frame["pressure_mean_0_3000"].to_numpy()),
        },
        coords={"observation": np.arange(len(frame))},
    )
    enriched = FeaturePipeline([Heterogeneity()])(data)
    return {
        name: float(np.asarray(value.values))
        for name, value in enriched.data_vars.items()
        if np.asarray(value.values).ndim == 0
    }

def metric_dict(metrics):
    return {
        getattr(key, "value", str(key)): float(value)
        for key, value in metrics.items()
    }

print("Project:", PROJECT)
print("Raw data:", RAW)
print("R1:", R1_PATH.name)
print("R3:", R3_PATH.name)

r3 = build_r3_frame()
r1 = build_r1_frame()

print("R3 frame:", r3.shape)
print("R1 frame:", r1.shape)

with open(OUTPUT / "campaign_heterogeneity.json", "w") as file:
    json.dump(heterogeneity_summary(r3), file, indent=2)

collocator = TemporalCollocator(
    satellite_time="time",
    airborne_time="time",
    tolerance=pd.Timedelta(seconds=30),
)

r3_for_collocation = r3.copy()
r1_for_collocation = r1.copy()

r3_for_collocation["time"] = pd.to_datetime(
    r3_for_collocation["time"],
    utc=True,
    errors="coerce",
)
r1_for_collocation["time"] = pd.to_datetime(
    r1_for_collocation["time"],
    utc=True,
    errors="coerce",
)

r3_for_collocation = r3_for_collocation.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
r1_for_collocation = r1_for_collocation.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

print(
    "R3 time:",
    r3_for_collocation["time"].min(),
    "->",
    r3_for_collocation["time"].max(),
)
print(
    "R1 time:",
    r1_for_collocation["time"].min(),
    "->",
    r1_for_collocation["time"].max(),
)

collocated = collocator.collocate(
    r3_for_collocation,
    r1_for_collocation,
)

if collocated.empty:
    left = r1_for_collocation.rename(columns={"time": "airborne_time"})
    right = r3_for_collocation.rename(columns={"time": "satellite_time"})

    collocated = pd.merge_asof(
        left.sort_values("airborne_time"),
        right.sort_values("satellite_time"),
        left_on="airborne_time",
        right_on="satellite_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=30),
        suffixes=("", "_satellite"),
    )

    collocated = collocated.dropna(
        subset=["satellite_time"]
    ).reset_index(drop=True)

    if not collocated.empty:
        collocated["temporal_difference"] = (
            collocated["airborne_time"] - collocated["satellite_time"]
        ).abs()
        collocated["temporal_difference_seconds"] = (
            collocated["temporal_difference"].dt.total_seconds()
        )

if collocated.empty:
    raise RuntimeError(
        "No R1/R3 temporal collocations found within 30 seconds. "
        f"R3 range={r3_for_collocation['time'].min()} -> {r3_for_collocation['time'].max()} | "
        f"R1 range={r1_for_collocation['time'].min()} -> {r1_for_collocation['time'].max()}"
    )

print("Temporal collocations:", len(collocated))
print(
    "Maximum temporal difference (s):",
    float(collocated["temporal_difference_seconds"].max()),
)

builder = ScientificBuilder()
builder.add(collocated)
dataset = builder.build()
dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.dropna(subset=["pbl_height_proxy_m"]).reset_index(drop=True)

target = "pbl_height_proxy_m"

excluded = {
    target,
    "time",
    "satellite_time",
    "airborne_time",
    "temporal_difference",
    "temporal_difference_seconds",
    "latitude",
    "longitude",
    "r1_latitude",
    "r1_longitude",
    "backscatter_gradient_min",
}

feature_names = [
    column
    for column in dataset.columns
    if column not in excluded
    and not column.startswith("backscatter_")
    and pd.api.types.is_numeric_dtype(dataset[column])
]

if not feature_names:
    raise RuntimeError(
        "No scientific features are available after removing target leakage and non-numeric columns."
    )

feature_frame = dataset[feature_names].replace([np.inf, -np.inf], np.nan)
feature_coverage = feature_frame.notna().mean().sort_values(ascending=False)
feature_names = feature_coverage[feature_coverage >= 0.50].index.tolist()

if not feature_names:
    raise RuntimeError(
        "No scientific feature has at least 50% valid coverage after collocation."
    )

feature_frame = dataset[feature_names].replace([np.inf, -np.inf], np.nan)

valid_target = dataset[target].notna().to_numpy()
valid_rows = np.where(valid_target)[0]

if len(valid_rows) < 30:
    raise RuntimeError(f"Too few valid scientific samples: {len(valid_rows)}")

splitter = Split(
    train=0.70,
    validation=0.15,
    test=0.15,
    shuffle=False,
)

train_idx, validation_idx, test_idx = splitter.build(len(valid_rows))

X_raw = feature_frame.to_numpy(dtype=np.float64)
y = dataset[target].to_numpy(dtype=np.float64)

train_feature_values = X_raw[train_idx]
medians = np.nanmedian(train_feature_values, axis=0)
medians = np.where(np.isfinite(medians), medians, 0.0)

X_raw = np.where(np.isfinite(X_raw), X_raw, medians)

scaler = Scaling(method=ScalingMethod.STANDARD)
X_train = scaler.fit_transform(X_raw[train_idx])
X_validation = scaler.transform(X_raw[validation_idx])
X_test = scaler.transform(X_raw[test_idx])

y_train = y[train_idx]
y_validation = y[validation_idx]
y_test = y[test_idx]

if len(y_train) == 0 or len(y_validation) == 0 or len(y_test) == 0:
    raise RuntimeError("Temporal split produced an empty train, validation, or test set.")

scientific_dataset = dataset.iloc[valid_rows].copy().reset_index(drop=True)
if "satellite_time" in scientific_dataset.columns:
    scientific_dataset["time"] = scientific_dataset["satellite_time"]
elif "airborne_time" in scientific_dataset.columns:
    scientific_dataset["time"] = scientific_dataset["airborne_time"]
else:
    raise KeyError("No collocated time column is available after temporal collocation.")
scientific_dataset.to_csv(OUTPUT / "scientific_dataset.csv", index=False)

netcdf_variables = {
    name: ("observation", scientific_dataset[name].to_numpy())
    for name in [target, "latitude", "longitude"] + feature_names
    if name in scientific_dataset.columns
}

xr.Dataset(
    netcdf_variables,
    coords={"observation": np.arange(len(scientific_dataset))},
    attrs={
        "campaign": "ACTIVATE",
        "date": "2022-01-11",
        "target": target,
        "target_definition": (
            "Proxy derived from the strongest negative vertical gradient of "
            "log10 cloud-screened 532 nm aerosol backscatter between 100 and 3000 m."
        ),
        "warning": "Research proxy, not an independently validated PBL height product.",
        "collocation": "Nearest-neighbour temporal collocation between R1 and R3 within 30 seconds.",
        "feature_policy": (
            "Backscatter-derived predictors excluded to avoid direct target leakage; "
            "remaining features with at least 50% valid coverage retained; "
            "training medians used for missing predictor values."
        ),
    },
).to_netcdf(OUTPUT / "pbl_scientific_dataset.nc")

models = {
    "linear": LinearRegressionModel(),
    "ridge": RidgeRegressionModel(alpha=1.0),
    "random_forest": RandomForestRegressionModel(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=12,
    ),
}

trained = {}
validation_metrics = {}
evaluator = RegressionEvaluator()

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    prediction = np.asarray(Predictor(model).predict(X_validation)).reshape(-1)
    validation_metrics[name] = metric_dict(
        evaluator.evaluate(y_validation, prediction)
    )
    trained[name] = model

best_name = min(
    validation_metrics,
    key=lambda name: validation_metrics[name]["rmse"],
)
best_model = trained[best_name]

test_prediction = np.asarray(
    Predictor(best_model).predict(X_test)
).reshape(-1)

test_metrics = metric_dict(
    evaluator.evaluate(y_test, test_prediction)
)

permutation_result = PermutationImportance(best_model).explain(
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
)

importance = np.asarray(
    permutation_result["importances_mean"]
).reshape(-1)

importance_frame = pd.DataFrame(
    {
        "feature": feature_names,
        "permutation_importance": importance,
    }
).sort_values(
    "permutation_importance",
    ascending=False,
)

importance_frame.to_csv(
    OUTPUT / "permutation_importance.csv",
    index=False,
)

def show_and_save_current(filename: str) -> None:
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
    )
    if display is not None:
        display(fig)
    else:
        plt.show()
    plt.close(fig)


print("\nDisplaying scientific figures...")

FeatureImportance().plot(
    importance,
    feature_names,
    top_k=min(15, len(feature_names)),
)
show_and_save_current("feature_importance.png")

validation_table = pd.DataFrame(
    [{"model": name, **metrics} for name, metrics in validation_metrics.items()]
)

TableExporter().export(
    validation_table,
    str(OUTPUT / "validation_metrics.csv"),
)

prediction_table = pd.DataFrame(
    {
        "time": scientific_dataset.iloc[test_idx]["time"].to_numpy(),
        "observed_pbl_height_proxy_m": y_test,
        "predicted_pbl_height_proxy_m": test_prediction,
        "residual_m": test_prediction - y_test,
    }
)

TableExporter().export(
    prediction_table,
    str(OUTPUT / "test_predictions.csv"),
)

ModelExporter().export(
    best_model,
    str(OUTPUT / "best_model.joblib"),
    ExportFormat.JOBLIB,
)

with open(OUTPUT / "metrics.json", "w") as file:
    json.dump(
        {
            "best_model": best_name,
            "validation": validation_metrics,
            "test": test_metrics,
            "n_samples": int(len(scientific_dataset)),
            "n_features": int(len(feature_names)),
            "features": feature_names,
            "split": {
                "train": int(len(train_idx)),
                "validation": int(len(validation_idx)),
                "test": int(len(test_idx)),
            },
        },
        file,
        indent=2,
    )

plotter = Plotter()

plotter.scatter(
    y_test,
    test_prediction,
)

plt.xlabel("Observed PBL-height proxy (m)")
plt.ylabel("Predicted PBL-height proxy (m)")
plt.title(f"{best_name}: temporal test")
show_and_save_current("prediction_vs_observation.png")

residuals = test_prediction - y_test

plotter.histogram(
    residuals,
    bins=30,
)

plt.xlabel("Residual (m)")
plt.ylabel("Frequency")
plt.title("Temporal test residuals")
show_and_save_current("residual_distribution.png")

TimeSeriesVisualizer().plot(
    scientific_dataset.iloc[test_idx]["time"].to_numpy(),
    y_test,
    label="Observed",
)

plt.plot(
    scientific_dataset.iloc[test_idx]["time"].to_numpy(),
    test_prediction,
    label="Predicted",
)

plt.xlabel("Time")
plt.ylabel("PBL-height proxy (m)")
plt.title("Temporal test")
plt.legend()
show_and_save_current("pbl_height_timeseries.png")

print("Scientific pipeline completed.")
print(f"Samples: {len(scientific_dataset)}")
print(f"Features: {len(feature_names)}")
print(f"Best model: {best_name}")
print(f"Validation metrics: {validation_metrics[best_name]}")
print(f"Test metrics: {test_metrics}")
print(f"Outputs: {OUTPUT}")
