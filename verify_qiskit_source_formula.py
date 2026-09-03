#!/usr/bin/env python3
"""Reproduce Qiskit's EffectiveDimension calculation with plain NumPy.

This file does not import Qiskit. It repeats the same normalization and log-determinant
steps used by qiskit_machine_learning.neural_networks.effective_dimension (0.9.1).
The inputs are exact constant Fisher matrices from the integer-shear example.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np


def logmeanexp(x):
    x=np.asarray(x,dtype=float); m=float(np.max(x))
    return m+math.log(float(np.mean(np.exp(x-m))))


def qiskit_source_equivalent_ed(fisher_samples, dataset_size):
    fisher_samples=np.asarray(fisher_samples,dtype=float)
    d=fisher_samples.shape[-1]
    fisher_trace=float(np.trace(np.average(fisher_samples,axis=0)))
    normalized=d*fisher_samples/fisher_trace
    k=dataset_size/(2*math.pi*math.log(dataset_size))
    mats=np.eye(d)[None,:,:]+k*normalized
    logdets=np.linalg.slogdet(mats)[1]
    return 2*logmeanexp(logdets/2)/math.log(k)


def torus_fisher(d,m):
    A=np.eye(d); A[0,1]=m
    return A.T@A


def analytic_ed(d,m,n):
    k=n/(2*math.pi*math.log(n)); t=k*d/(d+m*m)
    return ((d-2)*math.log1p(t)+math.log(1+t*(2+m*m)+t*t))/math.log(k)


def main():
    d=40; n=100_000; rows=[]
    for m in [0,1,5,10,20,100,1000,1_000_000]:
        exact=analytic_ed(d,m,n)
        # Repeating a constant Fisher matrix copies what any number of samples would give.
        fish=np.repeat(torus_fisher(d,m)[None,:,:],17,axis=0)
        mirror=qiskit_source_equivalent_ed(fish,n)
        rows.append({'d':d,'dataset_size':n,'integer_shear_m':m,'analytic_ed':exact,
                     'qiskit_source_equivalent_ed':mirror,'abs_difference':abs(mirror-exact)})
        print(f'm={m:>8g} exact={exact:.12f} mirror={mirror:.12f} diff={abs(mirror-exact):.3e}')
    with Path(__file__).with_name('qiskit_source_formula_check.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert max(r['abs_difference'] for r in rows)<1e-10
    print('Qiskit-source-equivalent functional check: PASS')

if __name__=='__main__': main()
