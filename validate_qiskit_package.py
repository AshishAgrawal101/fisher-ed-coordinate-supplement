#!/usr/bin/env python3
"""Check the exact ED formula with the real Qiskit Machine Learning 0.9.1 package.

This script does not simulate a QNN. Instead, it sends exact Fisher matrices through
Qiskit's public EffectiveDimension calculation. This catches package or API differences
that the plain-NumPy check might miss.
"""
import math
import numpy as np
from qiskit_machine_learning.neural_networks import EffectiveDimension

class DummyModel:
    def __init__(self, d):
        self.num_weights = d
        self.num_inputs = 1

class InjectedFisherED(EffectiveDimension):
    def __init__(self, raw_fisher):
        self._raw_fisher = np.asarray(raw_fisher, dtype=float)[None, :, :]
        d = self._raw_fisher.shape[-1]
        super().__init__(
            qnn=DummyModel(d),
            weight_samples=np.zeros((1, d)),
            input_samples=np.zeros((1, 1)),
        )

    def run_monte_carlo(self):
        # These arrays are placeholders; the next method supplies the exact Fisher matrix.
        return np.zeros((1, 1, self._model.num_weights)), np.ones((1, 1))

    def get_fisher_information(self, gradients, model_outputs):
        return self._raw_fisher.copy()


def analytic_ed(d, m, n):
    k=n/(2*math.pi*math.log(n))
    t=k*d/(d+m*m)
    return ((d-2)*math.log1p(t)+math.log(1+t*(2+m*m)+t*t))/math.log(k)

if __name__ == '__main__':
    import qiskit_machine_learning
    print('qiskit-machine-learning', qiskit_machine_learning.__version__)
    n=100_000; d=40
    for m in [0, 10, 100, 1000]:
        A=np.eye(d); A[0,1]=m
        F=A.T@A
        qval=float(InjectedFisherED(F).get_effective_dimension(n))
        aval=analytic_ed(d,m,n)
        print(f'm={m:4d} qiskit={qval:.12f} analytic={aval:.12f} diff={abs(qval-aval):.3e}')
        assert abs(qval-aval) < 1e-10
    print('Qiskit package-level functional validation: PASS')
