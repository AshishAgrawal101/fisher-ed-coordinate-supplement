#!/usr/bin/env python3
"""Check how the 2sED formula changes when we change parameter coordinates."""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np


def two_scale_from_eigs(eigs, epsilon, zeta):
    eigs=np.asarray(eigs,dtype=float); a=epsilon**(zeta-1)
    return zeta*len(eigs)+(1-zeta)*np.log1p(a*np.sqrt(eigs)).sum()/math.log(a)


def linear_spectrum(d,R):
    raw=np.array([R*R]+[1.0]*(d-1)); return d*raw/raw.sum()


def exact_linear_bounds(d,epsilon,zeta):
    a=epsilon**(zeta-1)
    lo=zeta*d+(1-zeta)*math.log(1+a*math.sqrt(d))/math.log(a)
    hi=zeta*d+(1-zeta)*d*math.log(1+a)/math.log(a)
    return lo,hi


def torus_integer_2sed(d,m,epsilon,zeta):
    a=epsilon**(zeta-1); c=d/(d+m*m); b=a*math.sqrt(c)
    logprod=(d-2)*math.log1p(b)+math.log(1+b*math.sqrt(m*m+4)+b*b)
    return zeta*d+(1-zeta)*logprod/math.log(a)


def main():
    d=40; n=100_000; epsilon=(math.log(n)/n)**(3/8); zeta=2/3
    lo,hi=exact_linear_bounds(d,epsilon,zeta)
    print(f'epsilon={epsilon:.12f}, zeta={zeta:.6f}')
    print(f'linear interval: ({lo:.9f}, {hi:.9f}]')
    rows=[]
    for R in [1,2,5,10,20,100,1000,1_000_000]:
        val=two_scale_from_eigs(linear_spectrum(d,R),epsilon,zeta)
        rows.append({'construction':'diagonal_linear','parameter':R,'two_scale_ed':val,'normalized':val/d})
    for m in [0,1,5,10,20,100,1000,1_000_000]:
        val=torus_integer_2sed(d,m,epsilon,zeta)
        rows.append({'construction':'same_torus_integer_shear','parameter':m,'two_scale_ed':val,'normalized':val/d})
    with Path(__file__).with_name('two_scale_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert abs(two_scale_from_eigs(np.ones(d),epsilon,zeta)-hi)<1e-10
    assert abs(torus_integer_2sed(d,100_000_000,epsilon,zeta)-lo)<1e-5
    print(f'same-torus m=1e8: {torus_integer_2sed(d,100_000_000,epsilon,zeta):.9f}')
    print('2sED coordinate-dependence checks: PASS')

if __name__=='__main__': main()
