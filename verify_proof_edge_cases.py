#!/usr/bin/env python3
"""Check the special cases that are easy to miss in the proofs."""
import math
import numpy as np


def ed_from_normalized_fisher(Fhat, kappa):
    sign, logdet = np.linalg.slogdet(np.eye(Fhat.shape[0]) + kappa * Fhat)
    assert sign > 0
    return logdet / math.log(kappa)


def normalized_constant_fisher(F):
    d = F.shape[0]
    return d * F / np.trace(F)


def torus_closed_form(d, m, kappa):
    c = d / (d + m*m)
    t = kappa * c
    return ((d-2)*math.log1p(t) + math.log(1 + t*(2+m*m) + t*t)) / math.log(kappa)


def two_scale_from_eigs(eigs, a, zeta):
    return zeta*len(eigs) + (1-zeta)/math.log(a) * sum(math.log(1+a*math.sqrt(x)) for x in eigs)


def regular_povm_fisher_numeric(r, ngrid=500_000):
    u = (np.arange(ngrid) + 0.5) * (2*np.pi/ngrid)
    val = r*r/(2*np.pi) * np.sum(np.sin(u)**2/(1+r*np.cos(u))) * (2*np.pi/ngrid)
    return float(val)


def random_spd(rng, d):
    M = rng.normal(size=(d,d))
    return M.T @ M + np.eye(d)


def main():
    rng = np.random.default_rng(20260831)
    kappa = 1382.402279

    # Compare the torus formula with a direct matrix calculation.
    for d in [2, 3, 8, 40]:
        for m in [0, 1, 5, 20, 1000]:
            A = np.eye(d)
            A[0,1] = m
            Fhat = normalized_constant_fisher(A.T @ A)
            direct = ed_from_normalized_fisher(Fhat, kappa)
            closed = torus_closed_form(d, m, kappa)
            assert abs(direct-closed) < 2e-11, (d,m,direct,closed)

    # With one parameter there is no ranking freedom; with two or more there is.
    for d in [1,2,3,10]:
        lo = math.log1p(kappa*d)/math.log(kappa)
        hi = d*math.log1p(kappa)/math.log(kappa)
        if d == 1:
            assert abs(lo-hi) < 1e-12
        else:
            assert hi > lo

    # Check the Fisher-information formula for the everywhere-positive POVM.
    for r in [0.1, 0.3, 0.5, 0.8, 0.95]:
        num = regular_povm_fisher_numeric(r)
        exact = 1-math.sqrt(1-r*r)
        assert abs(num-exact) < 2e-10, (r,num,exact)

    # Compare the 2sED torus formula with a direct eigenvalue calculation.
    a, zeta = 10.0, 0.2
    for d in [2,5,20]:
        for m in [0,2,10,1000]:
            A = np.eye(d)
            A[0,1] = m
            Fhat = normalized_constant_fisher(A.T @ A)
            eigs = np.linalg.eigvalsh(Fhat)
            direct = two_scale_from_eigs(eigs, a, zeta)
            c = d/(d+m*m)
            b = a*math.sqrt(c)
            closed = zeta*d + (1-zeta)/math.log(a) * (
                (d-2)*math.log1p(b) + math.log(1+b*math.sqrt(m*m+4)+b*b)
            )
            assert abs(direct-closed) < 1e-9, (d,m,direct,closed)

    # Check that the reference-metric version stays the same after a coordinate change.
    for d in [2,4,7]:
        for _ in range(20):
            F = random_spd(rng,d)
            G = random_spd(rng,d)
            J = rng.normal(size=(d,d))
            while abs(np.linalg.det(J)) < 0.1:
                J = rng.normal(size=(d,d))
            F2 = J.T @ F @ J
            G2 = J.T @ G @ J
            M1 = np.linalg.solve(G,F)
            M2 = np.linalg.solve(G2,F2)
            assert abs(np.trace(M1)-np.trace(M2)) < 1e-8
            for alpha in [0.01,1.0,100.0]:
                ld1 = np.linalg.slogdet(np.eye(d)+alpha*M1)[1]
                ld2 = np.linalg.slogdet(np.eye(d)+alpha*M2)[1]
                assert abs(ld1-ld2) < 1e-7

    print('proof edge-case checks: PASS')
    print('  torus closed form vs direct determinant: PASS')
    print('  d=1 ranking edge case: PASS')
    print('  strictly positive POVM Fisher identity: PASS')
    print('  2sED closed form vs direct eigenvalues: PASS')
    print('  reference-metric congruence invariance: PASS')

if __name__ == '__main__':
    main()
