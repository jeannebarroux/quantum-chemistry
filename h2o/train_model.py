"""Compare a small neural network with a quadratic fit to existing RHF data."""

from copy import deepcopy
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# Fixed before looking at test errors: no test-based hyperparameter search.
SEED = 42
LEARNING_RATE = 0.001
MAX_EPOCHS = 10000
PATIENCE = 500
MIN_DELTA = 1e-8  # Improvement in normalized validation MSE.


def load_data(path):
    """Check the CSV before selecting two inputs and one target."""
    names = ("bond_length_angstrom", "angle_deg", "energy_hartree")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.dtype.names != names or table.shape != (100,):
        raise ValueError("Expected exactly 100 rows and the three specified columns")
    data = np.column_stack([table[name] for name in names])
    if not np.isfinite(data).all():
        raise ValueError("Missing or non-finite data")
    x, y = data[:, :2], data[:, 2:3]
    if np.unique(x, axis=0).shape[0] != 100:
        raise ValueError("Duplicate geometries could leak across splits")
    return x, y


def split_indices(n):
    """Use one seeded permutation for both models: 70 train, 15 val, 15 test."""
    indices = np.random.default_rng(SEED).permutation(n)
    train, val, test = indices[:70], indices[70:85], indices[85:]
    if (len(train), len(val), len(test)) != (70, 15, 15):
        raise ValueError("Unexpected split sizes")
    if len(set(train) | set(val) | set(test)) != n:
        raise ValueError("Splits overlap or do not cover all rows")
    return train, val, test


def quadratic_features(x):
    """All degree <= 2 terms, including an intercept and the interaction."""
    r, angle = x[:, 0], x[:, 1]
    return np.column_stack([np.ones(len(x)), r, angle, r**2, r * angle, angle**2])


def as_tensor(array):
    """Convert NumPy data to a CPU tensor with double precision."""
    return torch.from_numpy(np.asarray(array, dtype=np.float64))


def train_network(x_train, y_train, x_val, y_val):
    """Fit on training data; choose and restore the best validation checkpoint."""
    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1),
    ).double()  # CPU float64 is inexpensive for only 100 geometries.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        # Both recorded losses describe the same post-update parameters.
        model.eval()
        with torch.no_grad():
            train_loss = loss_function(model(x_train), y_train).item()
            val_loss = loss_function(model(x_val), y_val).item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Non-finite training or validation loss")
        history.append((epoch, train_loss, val_loss))

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: train MSE={train_loss:.6g}, val MSE={val_loss:.6g}", flush=True)
        if stale_epochs >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, np.array(history), best_epoch


def errors(reference, prediction):
    """Inputs must be in hartree, after undoing target standardization."""
    if reference.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or non-finite predictions")
    residual = prediction - reference
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"mae_hartree": mae, "rmse_hartree": rmse,
            "mae_millihartree": 1000 * mae, "rmse_millihartree": 1000 * rmse}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(__file__).resolve().parent
    x, y = load_data(folder / "water_energies.csv")
    train, val, test = split_indices(len(x))

    # Never estimate preprocessing statistics from validation or test data.
    x_mean, x_std = x[train].mean(axis=0), x[train].std(axis=0)
    y_mean, y_std = float(y[train].mean()), float(y[train].std())
    if np.any(x_std <= 0) or y_std <= 0:
        raise ValueError("Cannot standardize a constant feature or target")
    x_train = (x[train] - x_mean) / x_std
    y_train = (y[train] - y_mean) / y_std
    model, history, best_epoch = train_network(
        as_tensor(x_train), as_tensor(y_train),
        as_tensor((x[val] - x_mean) / x_std), as_tensor((y[val] - y_mean) / y_std),
    )

    # Ordinary least squares, six coefficients; degree 2 is fixed in advance.
    design = quadratic_features(x_train)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y_train, rcond=None)
    if rank != 6:
        raise RuntimeError("Quadratic training design matrix is rank deficient")

    # First use of test values for prediction and evaluation, after fitting ends.
    x_test = (x[test] - x_mean) / x_std
    with torch.no_grad():
        neural_prediction = model(as_tensor(x_test)).numpy() * y_std + y_mean
    polynomial_prediction = quadratic_features(x_test) @ coefficients * y_std + y_mean
    metrics = {"neural_network": errors(y[test], neural_prediction),
               "quadratic": errors(y[test], polynomial_prediction)}
    for name, values in metrics.items():
        print(f"{name}: MAE={values['mae_hartree']:.10f} Eh "
              f"({values['mae_millihartree']:.6f} mEh), "
              f"RMSE={values['rmse_hartree']:.10f} Eh "
              f"({values['rmse_millihartree']:.6f} mEh)")
    print(f"Stopped after {len(history)} epochs; restored epoch {best_epoch}.")

    # Save the split, preprocessing and individual predictions for inspection.
    report = {
        "seed": SEED, "torch_version": torch.__version__, "numpy_version": np.__version__,
        "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "min_delta": MIN_DELTA,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "split_row_indices_zero_based": {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()},
        "training_statistics": {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                                "y_mean": y_mean, "y_std": y_std},
        "test_metrics": metrics,
    }
    (folder / "ml_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(folder / "test_predictions.csv",
               np.column_stack([test, x[test], y[test], neural_prediction, polynomial_prediction]),
               delimiter=",", fmt=["%d"] + ["%.12f"] * 5, comments="",
               header="row_index,bond_length_angstrom,angle_deg,rhf_energy_hartree,nn_energy_hartree,quadratic_energy_hartree")
    np.savetxt(folder / "training_history.csv", history, delimiter=",", comments="",
               header="epoch,training_mse_normalized,validation_mse_normalized", fmt=["%d", "%.12e", "%.12e"])

    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(history[:, 0], history[:, 1], label="Training")
    ax.semilogy(history[:, 0], history[:, 2], label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label="Restored epoch")
    ax.set(xlabel="Epoch", ylabel="MSE (standardized target)", title="Neural network learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y[test], neural_prediction, label="Neural network", marker="o")
    ax.scatter(y[test], polynomial_prediction, label="Quadratic", marker="x")
    limits = [min(y[test].min(), neural_prediction.min(), polynomial_prediction.min()),
              max(y[test].max(), neural_prediction.max(), polynomial_prediction.max())]
    ax.plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
    ax.set(xlabel="RHF energy (hartree)", ylabel="Predicted energy (hartree)",
           title="Held-out test geometries (15 points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "test_predictions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
