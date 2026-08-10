from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


SEED = 42
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

FEATURE_NAMES = [
    "AB_prfl",
    "cloud_height",
    "z",
]

OUTPUT_DIR = Path("technical_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


def validate_dataset(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.shape}")

    if X.shape[1] != 3:
        raise ValueError(f"Expected 3 features, got {X.shape[1]}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y have different lengths: {len(X)} != {len(y)}"
        )

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")

    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values")

    return X, y


class BaselineMLP(nn.Module):

    def __init__(self, n_features=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    y_val_scaled = y_scaler.transform(
        y_val.reshape(-1, 1)
    ).ravel()

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_scaled.reshape(-1, 1),
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val_scaled.reshape(-1, 1),
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor,
            y_train_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val_tensor,
            y_val_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BaselineMLP(
        n_features=X_train.shape[1]
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_history = []
    val_history = []

    for _ in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for xb, yb in train_loader:

            optimizer.zero_grad()

            prediction = model(xb)

            loss = criterion(
                prediction,
                yb,
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item() * len(xb)
            )

        train_loss /= len(X_train_tensor)

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for xb, yb in val_loader:

                prediction = model(xb)

                loss = criterion(
                    prediction,
                    yb,
                )

                validation_loss += (
                    loss.item() * len(xb)
                )

        validation_loss /= len(X_val_tensor)

        train_history.append(train_loss)
        val_history.append(validation_loss)

    return (
        model,
        x_scaler,
        y_scaler,
        train_history,
        val_history,
    )


def predict(
    model,
    x_scaler,
    y_scaler,
    X_data,
):
    X_scaled = x_scaler.transform(X_data)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():

        prediction_scaled = (
            model(X_tensor)
            .numpy()
            .ravel()
        )

    return y_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).ravel()


def evaluate(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def save_results(
    name,
    model,
    x_scaler,
    y_scaler,
    metrics,
    y_true,
    y_pred,
    train_history,
    val_history,
):

    directory = OUTPUT_DIR / name
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        directory / "model.pt",
    )

    joblib.dump(
        x_scaler,
        directory / "x_scaler.joblib",
    )

    joblib.dump(
        y_scaler,
        directory / "y_scaler.joblib",
    )

    with open(
        directory / "metrics.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    np.save(
        directory / "y_true.npy",
        y_true,
    )

    np.save(
        directory / "y_pred.npy",
        y_pred,
    )

    np.save(
        directory / "residuals.npy",
        y_true - y_pred,
    )

    np.save(
        directory / "train_history.npy",
        np.asarray(train_history),
    )

    np.save(
        directory / "val_history.npy",
        np.asarray(val_history),
    )

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.6,
    )

    lower = min(
        y_true.min(),
        y_pred.min(),
    )

    upper = max(
        y_true.max(),
        y_pred.max(),
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
    )

    plt.xlabel("Observed BSC 532")
    plt.ylabel("Predicted BSC 532")
    plt.title(
        f"{name} — Observed vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        directory / "observed_vs_predicted.png",
        dpi=300,
    )

    plt.close()

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_true,
        residuals,
        alpha=0.6,
    )

    plt.axhline(0)

    plt.xlabel("Observed BSC 532")
    plt.ylabel("Residual")
    plt.title(
        f"{name} — Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        directory / "residuals.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(train_history)
    plt.plot(val_history)

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        f"{name} — Training History"
    )

    plt.legend(
        ["Train", "Validation"]
    )

    plt.tight_layout()

    plt.savefig(
        directory / "training_history.png",
        dpi=300,
    )

    plt.close()


def run_temporal_split(X, y):

    n = len(y)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[
        train_end:validation_end
    ]

    y_val = y[
        train_end:validation_end
    ]

    X_test = X[
        validation_end:
    ]

    y_test = y[
        validation_end:
    ]

    (
        model,
        x_scaler,
        y_scaler,
        train_history,
        val_history,
    ) = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    y_pred = predict(
        model,
        x_scaler,
        y_scaler,
        X_test,
    )

    metrics = evaluate(
        y_test,
        y_pred,
    )

    metrics.update(
        {
            "split": "temporal",
            "random_state": None,
            "n_samples": int(len(y)),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
            "features": FEATURE_NAMES,
        }
    )

    save_results(
        "temporal_split",
        model,
        x_scaler,
        y_scaler,
        metrics,
        y_test,
        y_pred,
        train_history,
        val_history,
    )

    return metrics


def run_random_split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=SEED,
        shuffle=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15 / 0.85,
        random_state=SEED,
        shuffle=True,
    )

    (
        model,
        x_scaler,
        y_scaler,
        train_history,
        val_history,
    ) = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    y_pred = predict(
        model,
        x_scaler,
        y_scaler,
        X_test,
    )

    metrics = evaluate(
        y_test,
        y_pred,
    )

    metrics.update(
        {
            "split": "random",
            "random_state": SEED,
            "n_samples": int(len(y)),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
            "features": FEATURE_NAMES,
        }
    )

    save_results(
        "random_split",
        model,
        x_scaler,
        y_scaler,
        metrics,
        y_test,
        y_pred,
        train_history,
        val_history,
    )

    return metrics


def run_pipeline(X, y):

    X, y = validate_dataset(
        X,
        y,
    )

    random_metrics = run_random_split(
        X,
        y,
    )

    temporal_metrics = run_temporal_split(
        X,
        y,
    )

    summary = {
        "dataset": {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "features": FEATURE_NAMES,
            "target": "BSC_532",
        },
        "random_split": random_metrics,
        "temporal_split": temporal_metrics,
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print("=" * 80)
    print("TECHNICAL PIPELINE")
    print("=" * 80)

    print(
        f"Samples: {len(y)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "\nRandom split:"
    )

    for key in [
        "MAE",
        "RMSE",
        "R2",
    ]:
        print(
            f"{key}: "
            f"{random_metrics[key]:.8g}"
        )

    print(
        "\nTemporal split:"
    )

    for key in [
        "MAE",
        "RMSE",
        "R2",
    ]:
        print(
            f"{key}: "
            f"{temporal_metrics[key]:.8g}"
        )

    print(
        f"\nResults: {OUTPUT_DIR.resolve()}"
    )

    return summary


if __name__ == "__main__":

    input_path = Path(
        "technical_dataset.npz"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            "technical_dataset.npz not found. "
            "Create it with X and y from the validated "
            "technical dataset."
        )

    data = np.load(
        input_path
    )

    X = data["X"]
    y = data["y"]

    run_pipeline(
        X,
        y,
    )
