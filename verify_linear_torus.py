#!/usr/bin/env python3
import math
import csv
from pathlib import Path
import numpy as np


HERE = Path(__file__).resolve().parent


def kappa(n=100_000, gamma=1.0):
    return gamma*n/(2*math.pi*math.log(n))


def analytic_ed(d, m, n=100_000, gamma=1.0):
    k = kappa(n, gamma)
    t = k*d/(d+m*m)
    return ((d-2)*math.log1p(t) + math.log(1+t*(2+m*m)+t*t))/math.log(k)


def matrix_ed(d, m, n=100_000, gamma=1.0):
    k = kappa(n, gamma)
    A = np.eye(d)
    A[0,1] = m
    F = A.T @ A
    Fhat = d*F/np.trace(F)
    sign, logdet = np.linalg.slogdet(np.eye(d)+k*Fhat)
    assert sign > 0
    return logdet/math.log(k)


def infimum(d, n=100_000, gamma=1.0):
    k = kappa(n, gamma)
    return math.log1p(k*d)/math.log(k)


def paired_analytic_ratio(m, n=100_000, gamma=1.0):
    """ED/d for an even-dimensional torus with a shear on every pair."""
    k = kappa(n, gamma)
    t = 2*k/(2+m*m)
    return math.log(1+t*(2+m*m)+t*t)/(2*math.log(k))


def paired_matrix_ratio(d, m, n=100_000, gamma=1.0):
    assert d % 2 == 0
    k = kappa(n, gamma)
    A = np.eye(d)
    for index in range(0, d, 2):
        A[index, index+1] = m
    F = A.T @ A
    Fhat = d*F/np.trace(F)
    sign, logdet = np.linalg.slogdet(np.eye(d)+k*Fhat)
    assert sign > 0
    return logdet/(d*math.log(k))


def odd_paired_analytic_ratio(d, m, n=100_000, gamma=1.0):
    assert d % 2 == 1
    q = (d - 1) // 2
    k = kappa(n, gamma)
    t = k*d/(q*(2+m*m)+1)
    return (q*math.log(1+t*(2+m*m)+t*t)+math.log1p(t))/(d*math.log(k))


def odd_paired_matrix_ratio(d, m, n=100_000, gamma=1.0):
    assert d % 2 == 1
    k = kappa(n, gamma)
    A = np.eye(d)
    for index in range(0, d - 1, 2):
        A[index, index+1] = m
    Fhat = d*(A.T @ A)/np.trace(A.T @ A)
    sign, logdet = np.linalg.slogdet(np.eye(d)+k*Fhat)
    assert sign > 0
    return logdet/(d*math.log(k))

if __name__ == '__main__':
    d=40
    for m in [0,1,2,5,10,20,50,100,1000,1_000_000]:
        a=analytic_ed(d,m)
        b=matrix_ed(d,m)
        assert abs(a-b) < 1e-10
        print(f'm={m:>8g} ED={a:.9f} ED/d={a/d:.9f} matrix_diff={abs(a-b):.3e}')
    print('limit infimum:', infimum(d))
    assert abs(analytic_ed(d,1_000_000)-infimum(d)) < 1e-5
    rows = []
    for m in range(0, 11):
        closed = paired_analytic_ratio(m)
        ratios = [paired_matrix_ratio(dim, m) for dim in (2, 10, 40)]
        assert max(abs(value-closed) for value in ratios) < 1e-12
        block = np.array([[1.0, m], [0.0, 1.0]])
        condition = np.linalg.cond(block.T @ block)
        rows.append((m, closed, condition))
        print(f'paired m={m:2d} ED/d={closed:.9f} cond(A^T A)={condition:.3f}')
    with (HERE/'paired_torus_sensitivity.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('integer_shear_m', 'effective_dimension_over_d', 'condition_number_AtA'))
        writer.writerows(rows)
    for dim in (3, 9, 39):
        for m in (0, 2, 5, 10):
            closed = odd_paired_analytic_ratio(dim, m)
            direct = odd_paired_matrix_ratio(dim, m)
            assert abs(closed-direct) < 1e-12
    print('odd-dimensional paired formula: PASS')
    print('all linear-torus checks: PASS')
