#!/usr/bin/env python3
"""Rebuild the Abbas-style d=40 classical Fisher archive at high resolution.

The parameter vectors are sampled uniformly from [-1, 1]^40 and the four
inputs are independent standard Gaussians, matching the public implementation.
For two classes, the output Fisher has rank one, which lets us evaluate the
same analytic Fisher calculation without retaining every input contribution.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
D = 40
SEED = 0


def leaky_relu(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope = np.where(value >= 0.0, 1.0, 0.01)
    return np.where(value >= 0.0, value, 0.01 * value), slope


def rebuild(num_thetas: int, num_inputs: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    parameters = rng.uniform(-1.0, 1.0, size=(num_thetas, D))
    inputs = rng.normal(0.0, 1.0, size=(num_inputs, 4))
    raw_fisher = np.empty((num_thetas, D, D), dtype=np.float64)

    started = time.perf_counter()
    for index, theta in enumerate(parameters):
        w1 = theta[:16].reshape(4, 4)
        w2 = theta[16:32].reshape(4, 4)
        w3 = theta[32:].reshape(2, 4)

        z1 = inputs @ w1.T
        h1, slope1 = leaky_relu(z1)
        z2 = h1 @ w2.T
        h2, slope2 = leaky_relu(z2)
        z3 = h2 @ w3.T
        logits = np.tanh(z3)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        # J_0-J_1 is enough because diag(p)-pp^T = p0*p1 [[1,-1],[-1,1]].
        delta3 = np.column_stack(
            (1.0 - logits[:, 0] ** 2, -(1.0 - logits[:, 1] ** 2))
        )
        grad_w3 = np.einsum("ni,nj->nij", delta3, h2).reshape(num_inputs, 8)
        delta2 = (delta3 @ w3) * slope2
        grad_w2 = np.einsum("ni,nj->nij", delta2, h1).reshape(num_inputs, 16)
        delta1 = (delta2 @ w2) * slope1
        grad_w1 = np.einsum("ni,nj->nij", delta1, inputs).reshape(num_inputs, 16)
        gradient_difference = np.concatenate((grad_w1, grad_w2, grad_w3), axis=1)

        weights = probabilities[:, 0] * probabilities[:, 1]
        raw_fisher[index] = (
            gradient_difference.T @ (weights[:, None] * gradient_difference)
        ) / num_inputs

        if (index + 1) % 100 == 0 or index + 1 == num_thetas:
            elapsed = time.perf_counter() - started
            print(
                f"theta {index + 1}/{num_thetas}; elapsed={elapsed:.1f}s",
                flush=True,
            )

    mean_trace = float(np.trace(raw_fisher, axis1=1, axis2=2).mean())
    return D * raw_fisher / mean_trace


def main() -> None:
    protocol = rebuild(100, 100)
    old_protocol = np.load(HERE / "abbas_classical_d40_fhat.npy")
    maximum_error = float(np.max(np.abs(protocol - old_protocol)))
    if maximum_error > 2e-5:
        raise AssertionError(f"Vectorized Fisher does not match original: {maximum_error}")
    np.save(HERE / "abbas_classical_d40_protocol_100x100_fhat.npy", protocol)
    print(f"100x100 validation max abs error={maximum_error:.3e}", flush=True)

    theta_high = rebuild(2_000, 100)
    np.save(HERE / "abbas_classical_d40_2000x100_fhat.npy", theta_high)
    print("saved split 2000x100 Fisher archive", flush=True)

    input_high = rebuild(100, 2_000)
    np.save(HERE / "abbas_classical_d40_100x2000_fhat.npy", input_high)
    print("saved split 100x2000 Fisher archive", flush=True)

    high = rebuild(2_000, 2_000)
    np.save(HERE / "abbas_classical_d40_highres_2000x2000_fhat.npy", high)
    print("saved high-resolution 2000x2000 Fisher archive", flush=True)


if __name__ == "__main__":
    main()
