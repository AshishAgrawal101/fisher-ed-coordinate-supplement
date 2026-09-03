#!/usr/bin/env python3
"""Check that local ED matches global ED when the Fisher matrix is constant."""
import math
import numpy as np


def ed_from_fisher(F, n=100_000):
    d = F.shape[0]
    k = n / (2 * math.pi * math.log(n))
    Fhat = d * F / np.trace(F)
    sign, ld = np.linalg.slogdet(np.eye(d) + k * Fhat)
    assert sign > 0
    return ld / math.log(k)


def torus_closed_form(d, m, n=100_000):
    k = n / (2 * math.pi * math.log(n))
    t = k * d / (d + m * m)
    return ((d - 2) * math.log1p(t) + math.log(1 + t * (2 + m*m) + t*t)) / math.log(k)


if __name__ == '__main__':
    for d in [2, 5, 40]:
        for m in [0, 1, 5, 10, 100]:
            A = np.eye(d)
            A[0, 1] = m
            F = A.T @ A
            direct = ed_from_fisher(F)
            closed = torus_closed_form(d, m)
            diff = abs(direct - closed)
            print(f'd={d:2d} m={m:3d} local/direct={direct:.12f} closed={closed:.12f} diff={diff:.3e}')
            assert diff < 1e-11
    print('constant-Fisher local ED coordinate-dependence check: PASS')
