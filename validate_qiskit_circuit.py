#!/usr/bin/env python3
"""Run a Z-basis Qiskit circuit test for coordinate-dependent effective dimension.

The script builds two 2-qubit SamplerQNNs with the same family of output probabilities:
  theta-chart: Ry(theta_0) x Ry(theta_1)
  eta-chart:   Ry(eta_0 + m eta_1) x Ry(eta_1)
for an integer m. The map theta = A_m eta (mod 2*pi) is a smooth one-to-one
coordinate change on the torus. A Z-only measurement is not identifiable because
of angle-reflection symmetry; this test checks the Qiskit computation, not the
identifiability claim of the paper's X/Z construction.

It checks three things:
  1) matching parameter points give the same QNN probabilities;
  2) both SamplerQNNs run through Qiskit's EffectiveDimension code;
  3) Qiskit's answers match the exact constant-Fisher formula.

This file uses qiskit-machine-learning==0.9.1 and QMLSampler's exact mode.
"""
from __future__ import annotations

import math
import platform
import numpy as np
import qiskit
import qiskit_machine_learning
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.primitives import QMLSampler
from qiskit_machine_learning.neural_networks import SamplerQNN, EffectiveDimension


def analytic_ed(d: int, m: int, n: int) -> float:
    kappa = n / (2 * math.pi * math.log(n))
    t = kappa * d / (d + m * m)
    return ((d - 2) * math.log1p(t) + math.log(1 + t * (2 + m * m) + t * t)) / math.log(kappa)


def build_qnns(m: int):
    # Build the circuit in the original theta coordinates.
    th0, th1 = Parameter("theta0"), Parameter("theta1")
    qc_theta = QuantumCircuit(2)
    qc_theta.ry(th0, 0)
    qc_theta.ry(th1, 1)

    # Build the same circuit family in eta coordinates.
    et0, et1 = Parameter("eta0"), Parameter("eta1")
    qc_eta = QuantumCircuit(2)
    qc_eta.ry(et0 + m * et1, 0)
    qc_eta.ry(et1, 1)

    sampler = QMLSampler()  # QML 0.9.1 uses exact statevector mode here.
    qnn_theta = SamplerQNN(
        circuit=qc_theta,
        input_params=[],
        weight_params=[th0, th1],
        sampler=sampler,
    )
    qnn_eta = SamplerQNN(
        circuit=qc_eta,
        input_params=[],
        weight_params=[et0, et1],
        sampler=sampler,
    )
    return qnn_theta, qnn_eta


def mapped_theta(eta: np.ndarray, m: int) -> np.ndarray:
    theta = np.empty_like(eta, dtype=float)
    theta[:, 0] = np.mod(eta[:, 0] + m * eta[:, 1], 2 * np.pi)
    theta[:, 1] = np.mod(eta[:, 1], 2 * np.pi)
    return theta


def main() -> None:
    print("python", platform.python_version())
    print("numpy", np.__version__)
    print("qiskit", qiskit.__version__)
    print("qiskit-machine-learning", qiskit_machine_learning.__version__)

    assert qiskit_machine_learning.__version__ == "0.9.1", "Run with qiskit-machine-learning==0.9.1"

    n = 100_000
    d = 2
    m = 10
    qnn_theta, qnn_eta = build_qnns(m)

    # These four points give exact binary-fraction probabilities, so the test has no
    # Monte Carlo sampling or shot noise.
    vals = [np.pi / 2, 3 * np.pi / 2]
    eta_samples = np.array([(a, b) for a in vals for b in vals], dtype=float)
    theta_samples = mapped_theta(eta_samples, m)
    no_inputs = np.empty((1, 0), dtype=float)

    # First check that the two coordinate systems give the same probabilities.
    max_prob_diff = 0.0
    for eta, theta in zip(eta_samples, theta_samples):
        p_eta = np.asarray(qnn_eta.forward(input_data=no_inputs, weights=eta), dtype=float)
        p_theta = np.asarray(qnn_theta.forward(input_data=no_inputs, weights=theta), dtype=float)
        max_prob_diff = max(max_prob_diff, float(np.max(np.abs(p_eta - p_theta))))
    print(f"same-model max probability difference: {max_prob_diff:.3e}")

    ed_theta = EffectiveDimension(
        qnn=qnn_theta,
        weight_samples=theta_samples,
        input_samples=no_inputs,
    )
    ed_eta = EffectiveDimension(
        qnn=qnn_eta,
        weight_samples=eta_samples,
        input_samples=no_inputs,
    )

    val_theta = float(ed_theta.get_effective_dimension(n))
    val_eta = float(ed_eta.get_effective_dimension(n))
    expected_theta = analytic_ed(d=d, m=0, n=n)
    expected_eta = analytic_ed(d=d, m=m, n=n)

    print(f"theta chart: qiskit={val_theta:.12f} analytic={expected_theta:.12f} diff={abs(val_theta-expected_theta):.3e}")
    print(f"eta chart:   qiskit={val_eta:.12f} analytic={expected_eta:.12f} diff={abs(val_eta-expected_eta):.3e}")
    print(f"same model, ED shift: {val_theta:.12f} -> {val_eta:.12f} ({(val_eta/val_theta-1)*100:.2f}%)")

    assert max_prob_diff < 1e-12
    assert abs(val_theta - expected_theta) < 1e-9
    assert abs(val_eta - expected_eta) < 1e-9
    assert val_eta < val_theta - 0.5
    print("Qiskit circuit-level same-model validation: PASS")


if __name__ == "__main__":
    main()
