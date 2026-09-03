#!/usr/bin/env python3
"""Check local point-Fisher ED along a function-preserving ReLU rescaling."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


INPUTS = np.array(
    [
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0],
        [1.0, -1.0], [-1.0, -1.0], [2.0, 1.0], [1.0, 2.0],
        [-2.0, 1.0], [1.0, -2.0], [2.0, -1.0], [-1.0, 2.0],
    ],
    dtype=float,
)
BASE_W = np.array([[1.2, -0.7], [-0.4, 1.1]], dtype=float)
BASE_V = np.array([0.9, -1.3], dtype=float)
OUTPUT_BIAS = 0.2


def fisher_and_logits(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the Bernoulli Fisher matrix and logits after positive rescaling."""
    weights = alpha * BASE_W
    output_weights = BASE_V / alpha
    fisher_terms = []
    logits = []

    for x in INPUTS:
        preactivation = weights @ x
        hidden = np.maximum(preactivation, 0.0)
        logit = float(output_weights @ hidden + OUTPUT_BIAS)
        probability = 1.0 / (1.0 + math.exp(-logit))

        grad_weights = np.outer(output_weights * (preactivation > 0.0), x).ravel()
        gradient = np.concatenate((grad_weights, hidden, [1.0]))
        fisher_terms.append(probability * (1.0 - probability) * np.outer(gradient, gradient))
        logits.append(logit)

    return np.mean(fisher_terms, axis=0), np.asarray(logits)


def point_fisher_ed(fisher: np.ndarray, dataset_size: int) -> float:
    """Evaluate the trace-normalized determinant formula at one parameter point."""
    dimension = fisher.shape[0]
    kappa = dataset_size / (2.0 * math.pi * math.log(dataset_size))
    normalized = dimension * fisher / np.trace(fisher)
    sign, logdet = np.linalg.slogdet(np.eye(dimension) + kappa * normalized)
    assert sign > 0
    return float(logdet / math.log(kappa))


def main() -> None:
    dataset_size = 100_000
    base_fisher, base_logits = fisher_and_logits(alpha=1.0)
    rank = int(np.linalg.matrix_rank(base_fisher, tol=1e-10))
    rows = []

    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0]:
        fisher, logits = fisher_and_logits(alpha)
        max_logit_difference = float(np.max(np.abs(logits - base_logits)))
        effective_dimension = point_fisher_ed(fisher, dataset_size)
        rows.append(
            {
                "alpha": alpha,
                "max_logit_difference": max_logit_difference,
                "fisher_rank": int(np.linalg.matrix_rank(fisher, tol=1e-10)),
                "point_fisher_effective_dimension": effective_dimension,
            }
        )
        print(
            f"alpha={alpha:>5g} max_logit_diff={max_logit_difference:.3e} "
            f"rank={rows[-1]['fisher_rank']} ED={effective_dimension:.12f}"
        )

    output_path = Path(__file__).with_name("relu_local_ed_sensitivity.csv")
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    assert rank == 5
    assert max(row["max_logit_difference"] for row in rows) < 1e-12
    assert abs(rows[3]["point_fisher_effective_dimension"] - 4.790642183642922) < 1e-12
    assert abs(rows[6]["point_fisher_effective_dimension"] - 2.7547392292358217) < 1e-12
    print("function-preserving ReLU local-ED check: PASS")


if __name__ == "__main__":
    main()
