"""Observation noise, train-only normalization, and persistence utilities."""

from pathlib import Path
import json
import numpy as np


def make_dataset(time: np.ndarray, clean_states: np.ndarray, *,
                 observation_noise_std: float = 0.1,
                 train_fraction: float = 0.7,
                 seed: int = 42) -> dict[str, np.ndarray | int]:
    """Build a standardized synthetic observation dataset.

    Noise is Gaussian measurement noise, specified as a fraction of each
    channel's clean training standard deviation. Scaling uses only the
    observed training segment, preventing information leakage from the
    held-out segment. State-space comparison uses this shared scaling.
    This is observation noise, NOT the process noise used by the original paper.
    """
    time = np.asarray(time, dtype=float)
    clean_states = np.asarray(clean_states, dtype=float)
    if clean_states.ndim != 2 or time.ndim != 1 or len(time) != len(clean_states):
        raise ValueError("Expected time [T] and clean_states [T, D]")
    if len(time) < 10 or not (0.1 < train_fraction < 0.9):
        raise ValueError("Need >=10 observations and 0.1 < train_fraction < 0.9")
    if observation_noise_std < 0 or not np.isfinite(observation_noise_std):
        raise ValueError("observation_noise_std must be finite and nonnegative")
    if not np.isfinite(clean_states).all():
        raise ValueError("clean_states must contain only finite values")

    train_end = int(len(time) * train_fraction)
    clean_std = clean_states[:train_end].std(axis=0)
    if np.any(clean_std < 1e-12):
        raise ValueError("Cannot standardize a nearly constant channel")
    rng = np.random.default_rng(seed)
    measured = clean_states + rng.normal(
        0.0, observation_noise_std * clean_std,
        size=clean_states.shape,
    )
    mean = measured[:train_end].mean(axis=0)
    std = measured[:train_end].std(axis=0)
    if np.any(std < 1e-12):
        raise ValueError("No variation in an observed training channel")

    return {
        "time": time.astype(np.float64),
        "x_true": ((clean_states - mean) / std).astype(np.float32),
        "x_obs": ((measured - mean) / std).astype(np.float32),
        "train_mean": mean.astype(np.float64),
        "train_std": std.astype(np.float64),
        "train_end": train_end,
    }


def save_dataset(dataset: dict, path: str | Path, *, metadata: dict | None = None) -> None:
    """Save compressed arrays plus a human-readable JSON metadata file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    doc = {
        "notes": "Synthetic observed data with Gaussian observation noise; not process noise.",
        "n_steps": int(len(dataset["time"])),
        "n_dimensions": int(dataset["x_obs"].shape[1]),
        "train_end": int(dataset["train_end"]),
        **(metadata or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> dict:
    """Read a saved NPZ into memory."""
    with np.load(path, allow_pickle=False) as content:
        return {key: content[key] for key in content.files}
