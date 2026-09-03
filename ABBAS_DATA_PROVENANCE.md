# Abbas-model data provenance

The files `abbas_qnn_fhat.npy` and `abbas_easy_qnn_fhat.npy` are unchanged copies of
`4qubits_9layer_f_hats_dep2.npy` and `4qubits_9layer_f_hats_pauli.npy` from
<https://github.com/amyami187/effective_dimension>, commit
`5d9a9b638967bee5ff848c9564f4c90849afc5ca` (Apache-2.0 license). Their SHA-256 hashes are:

- QNN: `61A167DBD6B924B5904ACDD588FBDD94124F21200AD7BBFE2EB9C62DFBE31547`
- easy QNN: `DC531089C615256557B5A4183EAF0DEC03AE22204FB0207AE91BDE175E34AF3C`
- published classical archive: `1DFDC935105B8F57B7453925206F8A4BE4CABDA91F60C1CD7C41EA5BEFBBBAB8`

The quantum arrays have shape `(100, 40, 40)`. The repository's classical array is named
`fhat4_[4 4 4 2]_ed.npy`, but it has shape `(100, 28, 28)`, which conflicts with the stated
40-parameter `[4,4,4,2]` architecture. It exactly reproduces the published Figure 3a classical
source value and is retained under the clearer local name `abbas_published_classical_d28.npy`.
It is not treated as a 40-parameter model.

The build script separately reconstructs the stated 40-parameter leaky-ReLU/softmax model in
NumPy. Parameters are independent uniform draws on `[-1,1]^40`; inputs are independent standard
Gaussians in four dimensions; the seed is 0. It saves both the original 100-by-100 protocol and a
two split-count archives and a 2,000-parameter-by-2,000-input archive. Their SHA-256 hashes are:

- 100-by-100: `862AC45732C6FA3C2693B13CCBF2B922EE6DA6A2A4AF6BE81BA0EE827F42B86D`
- 2,000-by-100: `E836B6F7612DEE7EF1086B3F6ED57274D43B8287819B897FC774DDED54AEB6D2`
- 100-by-2,000: `DD39034420F3614CB9C991B338A8E15A47E686EF30E27ED21A8575CCDAE76030`
- 2,000-by-2,000: `379CBE0C9F462CD33E22047D9B0277586671DD287338ECDF3F5480BDAE175D46`

This is a transparent replication, not the missing original 40-parameter array. The coordinate
analysis applies one canonical and 49 seeded random oriented matchings to every quantum and
classical archive, using both shear signs and matched bounds on `cond(A.T @ A)`. For the d=28
archive each matching contains 14 pairs; for d=40 it contains 20. A separate 41-by-41 rescaling
grid covers `(log s1, log s2)` in `[-0.5 log(727), 0.5 log(727)]^2` for the rebuilt classical model.
Bootstrap seed 20260903 produces 2,000 replicates. Every replicate recomputes the average-trace
normalization. Basic intervals and bootstrap bias are reported for every selected endpoint.
Endpoint intervals condition on the full-sample transformation that attained each endpoint; the
selected transformations and all numerical outputs are included in CSV form.
