"""Shared perturbation sampling used by both one-vs-rest LIME and Fisher LIME.

Both surrogate-fitting methods must see the *same* perturbed neighborhood
around the explained instance so that any difference in their outputs is
attributable to the surrogate-fitting step alone, not to different noise
draws. This mirrors the problem statement: "摂動データは同じサンプリングとする".

The sampling scheme mirrors LIME's own defaults for continuous tabular data
(sample_around_instance=True): z = x + noise * feature_std, with an
exponential kernel of width sqrt(n_features) * 0.75 on the scaled distance.
"""
from __future__ import annotations

import numpy as np


def kernel_width_default(n_features: int) -> float:
    return float(np.sqrt(n_features) * 0.75)


def sample_perturbations(
    x: np.ndarray,
    feature_std: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    kernel_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample z_1..z_N around x and compute LIME-style kernel weights.

    Returns
    -------
    Z : (n_samples, n_features) perturbed points, with Z[0] == x (LIME
        always includes the original instance itself in the neighborhood).
    weights : (n_samples,) proximity weights pi_i in (0, 1].
    """
    n_features = x.shape[0]
    if kernel_width is None:
        kernel_width = kernel_width_default(n_features)

    noise = rng.standard_normal(size=(n_samples - 1, n_features))
    Z_rest = x[None, :] + noise * feature_std[None, :]
    Z = np.vstack([x[None, :], Z_rest])

    # distance in the same "number of stds away" space LIME uses internally
    scaled_diff = (Z - x[None, :]) / feature_std[None, :]
    dist = np.sqrt((scaled_diff ** 2).sum(axis=1))
    weights = np.sqrt(np.exp(-(dist ** 2) / (kernel_width ** 2)))
    return Z, weights
