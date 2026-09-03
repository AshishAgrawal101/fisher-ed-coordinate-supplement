# Fisher-Based Effective Dimension

This repository contains the code, saved Fisher-information arrays, provenance notes, and result tables for the accompanying manuscript. Everything needed to check the reported calculations is included here; no quantum hardware, GPU, external dataset, or model training is needed.

## Quick start

Use Python 3.14 and install the packages listed in `requirements.txt`.

```text
python verify_linear_torus.py
python verify_local_ed_coordinate_dependence.py
python verify_2sed_coordinate_dependence.py
python verify_qiskit_source_formula.py
python verify_proof_edge_cases.py
python verify_relu_local_ed.py
python verify_nonconstant_fisher_stretch.py
python build_classical_highres.py
python verify_abbas_coordinate_sensitivity.py
python validate_qiskit_package.py
python validate_qiskit_circuit.py
python make_figures.py
```

Each script either checks a result used in the manuscript or rebuilds a table or figure. The full console output from the checked version is saved in `verification_output_v14.txt`.

## What the files do

 `verify_linear_torus.py`, `verify_local_ed_coordinate_dependence.py`, and `verify_2sed_coordinate_dependence.py` check the closed-form coordinate-change results.
 `verify_nonconstant_fisher_stretch.py` checks the nonconstant-Fisher examples and the 2sED extension.
 `verify_relu_local_ed.py` shows the ReLU rescaling example.
 `verify_qiskit_source_formula.py`, `validate_qiskit_package.py`, and `validate_qiskit_circuit.py` check the small Qiskit example.
 `build_classical_highres.py` rebuilds the classical Fisher arrays at the listed sample sizes.
 `verify_abbas_coordinate_sensitivity.py` calculates the paired-shear, rescaling, bootstrap, and canonical-matching results.
 `make_figures.py` regenerates the figures from the result files.

## Main result files

 `abbas_matched_paired_shear_bands.csv`: paired-shear ranges for the compared models.
 `abbas_classical_rescaling_bands.csv`:  the classical ReLU-rescaling range.
 `abbas_native_resolution_and_bias.csv`:  results at four Fisher-estimation sample sizes.
 `abbas_bootstrap_contrasts.csv`:  basic-bootstrap intervals for the reported contrasts.
 `abbas_canonical_contrasts.csv`:  the matching-free robustness check.
 `abbas_pairing_robustness.csv`:  comparison of the canonical and multi-matching results.

## Data and provenance

The larger classical Fisher arrays are created locally by `build_classical_highres.py` instead of being kept in Git. `ABBAS_DATA_PROVENANCE.md` lists the source commit, array hashes, random seeds, sampling distributions, and generated-file hashes. The original upstream code license is saved in `ABBAS_CODE_LICENSE.txt`.
