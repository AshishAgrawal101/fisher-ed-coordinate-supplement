#!/usr/bin/env python3
"""Numerically verify the one-direction stretch lemma for a nonconstant Fisher field."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
D = 3
N = 100_000
KAPPA = N / (2.0 * np.pi * np.log(N))


def fisher_field(grid_size: int = 24) -> np.ndarray:
    axis = 2.0 * np.pi * (np.arange(grid_size) + 0.5) / grid_size
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.empty((len(points), D, D), dtype=float)
    for index, (x, y, z) in enumerate(points):
        b = np.array(
            [
                [1.0 + 0.35 * np.sin(x), 0.25 * np.cos(y), 0.10 * np.sin(z)],
                [0.20 * np.cos(x), 0.9 + 0.30 * np.sin(y), 0.20 * np.cos(z)],
                [0.15 * np.sin(x + y), 0.10 * np.cos(y + z), 0.7 + 0.25 * np.cos(z)],
            ]
        )
        values[index] = b.T @ b + 0.2 * np.eye(D)
    return values


def logmeanexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + float(np.log(np.mean(np.exp(values - peak))))


def effective_dimension(fisher: np.ndarray) -> float:
    normalized = D * fisher / np.trace(fisher, axis1=1, axis2=2).mean()
    logdets = np.linalg.slogdet(np.eye(D) + KAPPA * normalized)[1]
    return 2.0 * logmeanexp(logdets / 2.0) / np.log(KAPPA)


def stretched(fisher: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    jacobian = np.eye(D) + (scale - 1.0) * np.outer(direction, direction)
    return np.einsum("ji,njk,kl->nil", jacobian, fisher, jacobian)


def limiting_value(fisher: np.ndarray, direction: np.ndarray) -> float:
    direction = direction / np.linalg.norm(direction)
    directional = np.einsum("i,nij,j->n", direction, fisher, direction)
    ratio = directional / directional.mean()
    return 2.0 * np.log(np.mean(np.sqrt(1.0 + KAPPA * D * ratio))) / np.log(KAPPA)


def two_scale_ed(fisher: np.ndarray, epsilon: float, zeta: float) -> float:
    normalized = D * fisher / np.trace(fisher, axis1=1, axis2=2).mean()
    eigenvalues = np.maximum(np.linalg.eigvalsh(normalized), 0.0)
    a = epsilon ** (zeta - 1.0)
    determinants = np.prod(1.0 + a * np.sqrt(eigenvalues), axis=1)
    return zeta * D + (1.0 - zeta) * np.log(determinants.mean()) / np.log(a)


def limiting_two_scale(
    fisher: np.ndarray, direction: np.ndarray, epsilon: float, zeta: float
) -> tuple[float, float]:
    direction = direction / np.linalg.norm(direction)
    directional = np.einsum("i,nij,j->n", direction, fisher, direction)
    ratio = directional / directional.mean()
    a = epsilon ** (zeta - 1.0)
    limit = zeta * D + (1.0 - zeta) * np.log(
        np.mean(1.0 + a * np.sqrt(D * ratio))
    ) / np.log(a)
    bound = zeta * D + (1.0 - zeta) * np.log(1.0 + a * np.sqrt(D)) / np.log(a)
    return float(limit), float(bound)


def main() -> None:
    fisher = fisher_field()
    direction = np.array([1.0, 0.0, 0.0])
    limit = limiting_value(fisher, direction)
    constant_fisher_bound = np.log(1.0 + KAPPA * D) / np.log(KAPPA)
    rows = []
    for scale in (1.0, 3.0, 10.0, 100.0, 10_000.0, 1_000_000.0):
        value = effective_dimension(stretched(fisher, direction, scale))
        rows.append((scale, value))
        print(f"scale={scale:8g} ED={value:.12f}")
    print(f"directional limit={limit:.12f}")
    print(f"universal Jensen bound={constant_fisher_bound:.12f}")

    if abs(rows[-1][1] - limit) > 2e-7:
        raise AssertionError("Finite stretch did not converge to the directional formula")
    if not limit < constant_fisher_bound - 1e-8:
        raise AssertionError("Strict Jensen bound failed for the nonconstant directional Fisher")
    if not rows[0][1] > limit:
        raise AssertionError("Example does not display the intended decrease")

    epsilon = (math.log(N) / N) ** (3.0 / 8.0)
    zeta = 2.0 / 3.0
    two_scale_limit, two_scale_bound = limiting_two_scale(
        fisher, direction, epsilon, zeta
    )
    finite_two_scale = two_scale_ed(
        stretched(fisher, direction, 100_000_000.0), epsilon, zeta
    )
    if abs(finite_two_scale - two_scale_limit) > 2e-7:
        raise AssertionError("2sED stretch did not converge to the rank-one formula")
    if not two_scale_limit < two_scale_bound:
        raise AssertionError("Strict nonconstant-Fisher 2sED Jensen bound failed")
    print(f"2sED directional limit={two_scale_limit:.12f}")
    print(f"2sED Jensen bound={two_scale_bound:.12f}")

    with (HERE / "nonconstant_fisher_stretch.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stretch_scale", "effective_dimension"))
        writer.writerows(rows)
        writer.writerow(("directional_limit", limit))
        writer.writerow(("universal_jensen_bound", constant_fisher_bound))
    with (HERE / "nonconstant_fisher_2sed.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("quantity", "two_scale_effective_dimension"))
        writer.writerow(("finite_stretch_1e8", finite_two_scale))
        writer.writerow(("directional_limit", two_scale_limit))
        writer.writerow(("jensen_bound", two_scale_bound))
    print("nonconstant-Fisher stretch check: PASS")


if __name__ == "__main__":
    main()
