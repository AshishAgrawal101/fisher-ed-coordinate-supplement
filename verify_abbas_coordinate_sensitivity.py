#!/usr/bin/env python3
"""Matched-family coordinate-sensitivity checks for the Abbas et al. models.

All four Fisher archives use the same paired-shear family: one canonical and
49 seeded random oriented perfect matchings, both shear signs, and |m| <= 5.
The rebuilt d=40 classical archive is also evaluated on its exact two-layer
function-preserving rescaling orbit as a separate analysis.

Basic bootstrap intervals correct the visible first-order shift of percentile
intervals. Every replicate recomputes the average-trace normalization. Endpoint
intervals condition on the full-sample transformation selected for that endpoint.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
N_DATA_SCALE = 100_000
PAIRING_SEED = 20260902
BOOTSTRAP_SEED = 20260903
BOOTSTRAPS = 2_000
NUM_PAIRINGS = 50
RESCALING_GRID_SIZE = 41
M_VALUES = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Candidate:
    label: str
    jacobian: np.ndarray
    condition: float
    value: float


def logmeanexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    peak = np.max(values, axis=axis, keepdims=True)
    result = peak + np.log(np.mean(np.exp(values - peak), axis=axis, keepdims=True))
    result = np.squeeze(result, axis=axis) if axis is not None else np.squeeze(result)
    return float(result) if np.ndim(result) == 0 else result


def effective_dimension(fhat: np.ndarray, n: int = N_DATA_SCALE) -> float:
    d = fhat.shape[1]
    kappa = n / (2.0 * np.pi * np.log(n))
    logdets = np.linalg.slogdet(np.eye(d) + kappa * fhat)[1]
    return 2.0 * float(logmeanexp(logdets / 2.0)) / np.log(kappa)


def transform_and_normalize(fisher: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    transformed = np.einsum("ji,njk,kl->nil", jacobian, fisher, jacobian, optimize=True)
    mean_trace = float(np.trace(transformed, axis1=1, axis2=2).mean())
    return fisher.shape[1] * transformed / mean_trace


def ed_for_transform(fisher: np.ndarray, jacobian: np.ndarray) -> float:
    return effective_dimension(transform_and_normalize(fisher, jacobian)) / fisher.shape[1]


def matrix_condition(jacobian: np.ndarray) -> float:
    return float(np.linalg.cond(jacobian.T @ jacobian))


def shear_condition(magnitude: int) -> float:
    block = np.array([[1.0, float(magnitude)], [0.0, 1.0]])
    return matrix_condition(block)


def make_pairings(dimension: int) -> list[np.ndarray]:
    if dimension % 2:
        raise ValueError("Paired shears require an even dimension")
    pairings = [np.arange(dimension, dtype=int).reshape(-1, 2)]
    rng = np.random.default_rng(PAIRING_SEED + dimension)
    seen = {tuple(pairings[0].ravel())}
    while len(pairings) < NUM_PAIRINGS:
        pairing = rng.permutation(dimension).reshape(-1, 2)
        key = tuple(pairing.ravel())
        if key not in seen:
            pairings.append(pairing)
            seen.add(key)
    return pairings


def paired_shear(dimension: int, pairing: np.ndarray, signed_m: int) -> np.ndarray:
    jacobian = np.eye(dimension)
    for target, source in pairing:
        jacobian[target, source] = signed_m
    return jacobian


def paired_shear_candidates(fhat: np.ndarray) -> list[Candidate]:
    dimension = fhat.shape[1]
    identity = np.eye(dimension)
    candidates = [Candidate("identity", identity, 1.0, ed_for_transform(fhat, identity))]
    for pairing_index, pairing in enumerate(make_pairings(dimension)):
        for signed_m in range(-5, 6):
            if signed_m == 0:
                continue
            jacobian = paired_shear(dimension, pairing, signed_m)
            candidates.append(
                Candidate(
                    f"pairing={pairing_index};m={signed_m:+d}",
                    jacobian,
                    matrix_condition(jacobian),
                    ed_for_transform(fhat, jacobian),
                )
            )
    return candidates


def layer_rescaling_from_logs(a: float, b: float) -> np.ndarray:
    diagonal = np.concatenate(
        (
            np.full(16, np.exp(a)),
            np.full(16, np.exp(b - a)),
            np.full(8, np.exp(-b)),
        )
    )
    return np.diag(diagonal)


def rescaling_candidates(fhat: np.ndarray, max_condition: float) -> list[Candidate]:
    half_log_condition = 0.5 * np.log(max_condition)
    grid = np.linspace(-half_log_condition, half_log_condition, RESCALING_GRID_SIZE)
    candidates: list[Candidate] = []
    for a, b in itertools.product(grid, repeat=2):
        jacobian = layer_rescaling_from_logs(float(a), float(b))
        condition = matrix_condition(jacobian)
        if condition <= max_condition * (1.0 + 1e-10):
            candidates.append(
                Candidate(
                    f"log_s1={a:.12g};log_s2={b:.12g}",
                    jacobian,
                    condition,
                    ed_for_transform(fhat, jacobian),
                )
            )
    return candidates


def endpoint(candidates: list[Candidate], condition_bound: float, side: str) -> Candidate:
    allowed = [item for item in candidates if item.condition <= condition_bound * (1 + 1e-10)]
    if not allowed:
        raise AssertionError("No transformation satisfies condition bound")
    return (min if side == "low" else max)(allowed, key=lambda item: item.value)


def fixed_transform_bootstrap(
    fhat: np.ndarray, jacobian: np.ndarray, bootstrap_indices: np.ndarray
) -> np.ndarray:
    """Resample Fisher matrices and renormalize the mean trace in each replicate."""
    dimension = fhat.shape[1]
    transformed = np.einsum("ji,njk,kl->nil", jacobian, fhat, jacobian, optimize=True)
    traces = np.trace(transformed, axis1=1, axis2=2)
    eigenvalues = np.maximum(np.linalg.eigvalsh(transformed), 0.0)
    kappa = N_DATA_SCALE / (2.0 * np.pi * np.log(N_DATA_SCALE))
    results = np.empty(bootstrap_indices.shape[0], dtype=float)
    for start in range(0, len(results), 50):
        stop = min(start + 50, len(results))
        indices = bootstrap_indices[start:stop]
        sampled_traces = traces[indices].mean(axis=1)
        scales = dimension / sampled_traces
        sampled_eigenvalues = eigenvalues[indices]
        logdets = np.log1p(
            kappa * scales[:, None, None] * sampled_eigenvalues
        ).sum(axis=2)
        results[start:stop] = 2.0 * logmeanexp(logdets / 2.0, axis=1) / (
            dimension * np.log(kappa)
        )
    return results


def basic_interval(point: float, samples: np.ndarray) -> tuple[float, float]:
    q_low, q_high = np.quantile(samples, [0.025, 0.975])
    return float(2 * point - q_high), float(2 * point - q_low)


def selected_rows(
    model: str,
    fhat: np.ndarray,
    candidates: list[Candidate],
    condition_bounds: dict[int, float],
    bootstrap_indices: np.ndarray,
    family: str,
) -> tuple[list[dict[str, object]], dict[tuple[int, str], tuple[Candidate, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    selected: dict[tuple[int, str], tuple[Candidate, np.ndarray]] = {}
    for m, bound in condition_bounds.items():
        for side in ("low", "high"):
            candidate = endpoint(candidates, bound, side)
            samples = fixed_transform_bootstrap(fhat, candidate.jacobian, bootstrap_indices)
            ci_low, ci_high = basic_interval(candidate.value, samples)
            bias = float(np.mean(samples) - candidate.value)
            rows.append(
                {
                    "model": model,
                    "dimension": fhat.shape[1],
                    "family": family,
                    "shear_m_or_matched_budget": m,
                    "condition_bound": bound,
                    "endpoint": side,
                    "estimate": candidate.value,
                    "bootstrap_mean": float(np.mean(samples)),
                    "bootstrap_bias": bias,
                    "bootstrap_standard_error": float(np.std(samples, ddof=1)),
                    "basic_ci_2.5_percent": ci_low,
                    "basic_ci_97.5_percent": ci_high,
                    "attaining_transform": candidate.label,
                    "attaining_condition": candidate.condition,
                    "bootstrap_replicates": len(samples),
                    "trace_renormalized_each_replicate": True,
                }
            )
            selected[(m, side)] = (candidate, samples)
    return rows, selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data_sets = {
        "QNN": np.load(HERE / "abbas_qnn_fhat.npy"),
        "easy QNN": np.load(HERE / "abbas_easy_qnn_fhat.npy"),
        "published classical archive": np.load(HERE / "abbas_published_classical_d28.npy"),
        "rebuilt classical 100x100": np.load(
            HERE / "abbas_classical_d40_protocol_100x100_fhat.npy"
        ),
        "rebuilt classical 2000x100": np.load(
            HERE / "abbas_classical_d40_2000x100_fhat.npy"
        ),
        "rebuilt classical 100x2000": np.load(
            HERE / "abbas_classical_d40_100x2000_fhat.npy"
        ),
        "rebuilt classical 2000x2000": np.load(
            HERE / "abbas_classical_d40_highres_2000x2000_fhat.npy"
        ),
    }
    expected = {
        "QNN": (100, 40, 40),
        "easy QNN": (100, 40, 40),
        "published classical archive": (100, 28, 28),
        "rebuilt classical 100x100": (100, 40, 40),
        "rebuilt classical 2000x100": (2000, 40, 40),
        "rebuilt classical 100x2000": (100, 40, 40),
        "rebuilt classical 2000x2000": (2000, 40, 40),
    }
    for model, shape in expected.items():
        if data_sets[model].shape != shape:
            raise AssertionError(f"Unexpected {model} shape: {data_sets[model].shape}")

    condition_bounds = {m: shear_condition(m) for m in M_VALUES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = {
        model: rng.integers(0, array.shape[0], size=(BOOTSTRAPS, array.shape[0]))
        for model, array in data_sets.items()
    }

    paired_models = (
        "QNN",
        "easy QNN",
        "published classical archive",
        "rebuilt classical 2000x2000",
    )
    paired_rows: list[dict[str, object]] = []
    paired_selected: dict[str, dict[tuple[int, str], tuple[Candidate, np.ndarray]]] = {}
    candidate_sets: dict[str, list[Candidate]] = {}
    for model in paired_models:
        print(f"evaluating paired-shear candidates: {model}", flush=True)
        candidates = paired_shear_candidates(data_sets[model])
        candidate_sets[model] = candidates
        rows, selected = selected_rows(
            model,
            data_sets[model],
            candidates,
            condition_bounds,
            bootstrap_indices[model],
            "50 paired matchings, both signs",
        )
        paired_rows.extend(rows)
        paired_selected[model] = selected
    write_csv(HERE / "abbas_matched_paired_shear_bands.csv", paired_rows)

    high_classical = data_sets["rebuilt classical 2000x2000"]
    max_condition = condition_bounds[5]
    print("evaluating classical rescaling grid", flush=True)
    rescaling = rescaling_candidates(high_classical, max_condition)
    rescaling_rows, rescaling_selected = selected_rows(
        "rebuilt classical 2000x2000",
        high_classical,
        rescaling,
        condition_bounds,
        bootstrap_indices["rebuilt classical 2000x2000"],
        "41x41 function-preserving rescaling grid",
    )
    write_csv(HERE / "abbas_classical_rescaling_bands.csv", rescaling_rows)

    input_counts = {
        "QNN": 100,
        "easy QNN": 100,
        "published classical archive": 100,
        "rebuilt classical 100x100": 100,
        "rebuilt classical 2000x100": 100,
        "rebuilt classical 100x2000": 2000,
        "rebuilt classical 2000x2000": 2000,
    }
    native_rows: list[dict[str, object]] = []
    for model in data_sets:
        fhat = data_sets[model]
        point = effective_dimension(fhat) / fhat.shape[1]
        samples = fixed_transform_bootstrap(
            fhat, np.eye(fhat.shape[1]), bootstrap_indices[model]
        )
        ci_low, ci_high = basic_interval(point, samples)
        native_rows.append(
            {
                "model": model,
                "dimension": fhat.shape[1],
                "parameter_samples": fhat.shape[0],
                "input_samples": input_counts[model],
                "estimate": point,
                "bootstrap_mean": float(np.mean(samples)),
                "bootstrap_bias": float(np.mean(samples) - point),
                "bootstrap_standard_error": float(np.std(samples, ddof=1)),
                "basic_ci_2.5_percent": ci_low,
                "basic_ci_97.5_percent": ci_high,
                "trace_renormalized_each_replicate": True,
            }
        )
    write_csv(HERE / "abbas_native_resolution_and_bias.csv", native_rows)

    contrasts: list[dict[str, object]] = []
    for quantum_model in ("QNN", "easy QNN"):
        for m in (2, 3, 4, 5):
            q_candidate, q_samples = paired_selected[quantum_model][(m, "low")]
            c_candidate, c_samples = paired_selected["rebuilt classical 2000x2000"][(m, "high")]
            point = c_candidate.value - q_candidate.value
            samples = c_samples - q_samples
            ci_low, ci_high = basic_interval(point, samples)
            contrasts.append(
                {
                    "contrast": f"rebuilt classical high minus {quantum_model} low",
                    "m": m,
                    "condition_bound": condition_bounds[m],
                    "estimate": point,
                    "bootstrap_mean": float(np.mean(samples)),
                    "bootstrap_bias": float(np.mean(samples) - point),
                    "bootstrap_standard_error": float(np.std(samples, ddof=1)),
                    "basic_ci_2.5_percent": ci_low,
                    "basic_ci_97.5_percent": ci_high,
                }
            )
    write_csv(HERE / "abbas_bootstrap_contrasts.csv", contrasts)

    canonical_contrasts: list[dict[str, object]] = []
    for quantum_model in ("QNN", "easy QNN"):
        q_candidates = [
            item for item in candidate_sets[quantum_model]
            if item.label == "identity" or item.label.startswith("pairing=0;")
        ]
        c_candidates = [
            item for item in candidate_sets["rebuilt classical 2000x2000"]
            if item.label == "identity" or item.label.startswith("pairing=0;")
        ]
        q_candidate = endpoint(q_candidates, condition_bounds[5], "low")
        c_candidate = endpoint(c_candidates, condition_bounds[5], "high")
        q_samples = fixed_transform_bootstrap(
            data_sets[quantum_model], q_candidate.jacobian, bootstrap_indices[quantum_model]
        )
        c_samples = fixed_transform_bootstrap(
            high_classical,
            c_candidate.jacobian,
            bootstrap_indices["rebuilt classical 2000x2000"],
        )
        point = c_candidate.value - q_candidate.value
        samples = c_samples - q_samples
        ci_low, ci_high = basic_interval(point, samples)
        canonical_contrasts.append(
            {
                "contrast": f"rebuilt classical high minus {quantum_model} low",
                "matching": "canonical only",
                "m": 5,
                "estimate": point,
                "bootstrap_bias": float(np.mean(samples) - point),
                "basic_ci_2.5_percent": ci_low,
                "basic_ci_97.5_percent": ci_high,
            }
        )
    write_csv(HERE / "abbas_canonical_contrasts.csv", canonical_contrasts)

    robustness: list[dict[str, object]] = []
    for model in paired_models:
        all_candidates = candidate_sets[model]
        canonical = [
            item
            for item in all_candidates
            if item.label == "identity" or item.label.startswith("pairing=0;")
        ]
        for side in ("low", "high"):
            all_endpoint = endpoint(all_candidates, condition_bounds[5], side)
            canonical_endpoint = endpoint(canonical, condition_bounds[5], side)
            robustness.append(
                {
                    "model": model,
                    "endpoint": side,
                    "canonical_pairing_only": canonical_endpoint.value,
                    "fifty_matchings": all_endpoint.value,
                    "difference": all_endpoint.value - canonical_endpoint.value,
                }
            )
    write_csv(HERE / "abbas_pairing_robustness.csv", robustness)

    details: list[dict[str, object]] = []
    for model, candidates in candidate_sets.items():
        for candidate in candidates:
            details.append(
                {
                    "model": model,
                    "transform": candidate.label,
                    "condition": candidate.condition,
                    "ed_over_d": candidate.value,
                }
            )
    write_csv(HERE / "abbas_paired_shear_family_details.csv", details)

    protocol_native = effective_dimension(data_sets["rebuilt classical 100x100"]) / 40
    high_native = effective_dimension(data_sets["rebuilt classical 2000x2000"]) / 40
    if high_native - protocol_native < 0.04:
        raise AssertionError("High-resolution rebuild did not expose the low-sample shift")
    qnn_m3 = next(
        row for row in contrasts if row["contrast"].endswith("QNN low") and row["m"] == 3
    )
    if qnn_m3["basic_ci_2.5_percent"] <= 0:
        raise AssertionError("Corrected QNN/classical contrast is not supported at m=3")
    if canonical_contrasts[0]["basic_ci_2.5_percent"] <= 0:
        raise AssertionError("Canonical-only QNN/classical contrast is not supported")
    if any(len(candidate_sets[model]) != 501 for model in paired_models):
        raise AssertionError("A paired-shear search did not use the full shared budget")
    if not all(np.isfinite(float(row["estimate"])) for row in paired_rows + rescaling_rows):
        raise AssertionError("Non-finite endpoint estimate")

    print(f"pairing seed={PAIRING_SEED}; bootstrap seed={BOOTSTRAP_SEED}")
    print(f"pairings={NUM_PAIRINGS}; rescaling grid={RESCALING_GRID_SIZE}x{RESCALING_GRID_SIZE}")
    for row in native_rows:
        print(
            f"native {row['model']}: {row['estimate']:.6f}, "
            f"bias={row['bootstrap_bias']:+.6f}, "
            f"basic CI=[{row['basic_ci_2.5_percent']:.6f},{row['basic_ci_97.5_percent']:.6f}]"
        )
    for row in paired_rows:
        if row["shear_m_or_matched_budget"] == 5:
            print(
                f"paired {row['model']:31s} {row['endpoint']:4s}: {row['estimate']:.6f}, "
                f"bias={row['bootstrap_bias']:+.6f}, "
                f"CI=[{row['basic_ci_2.5_percent']:.6f},{row['basic_ci_97.5_percent']:.6f}]"
            )
    for row in contrasts:
        if row["m"] == 5:
            print(
                f"{row['contrast']}: {row['estimate']:+.6f}, "
                f"bias={row['bootstrap_bias']:+.6f}, "
                f"CI=[{row['basic_ci_2.5_percent']:.6f},{row['basic_ci_97.5_percent']:.6f}]"
            )
    for row in canonical_contrasts:
        print(
            f"canonical-only {row['contrast']}: {row['estimate']:+.6f}, "
            f"CI=[{row['basic_ci_2.5_percent']:.6f},{row['basic_ci_97.5_percent']:.6f}]"
        )
    print(
        "rescaling m=5-equivalent band: "
        f"[{rescaling_selected[(5, 'low')][0].value:.6f},"
        f"{rescaling_selected[(5, 'high')][0].value:.6f}]"
    )
    print("matched-family bias-aware coordinate analysis: PASS")


if __name__ == "__main__":
    main()
