import sympy as sp

from qudit_depolarization_channels import *
from qudit_gates import *


class QAOAQuditAnsatz:
    def __init__(self, num_qudits=2, num_layers=1, pairing_prob=0.4, with_noise=False):

        self.gamma = sp.symbols("γ0:num_layers")
        self.beta = sp.symbols("β0:num_layers")
        self.num_qudits = num_qudits
        self.num_layers = num_layers
        self.pairing_prob = pairing_prob
        self.with_noise = with_noise
        self.qudits = cirq.LineQid.range(num_qudits, dimension=4)
        self.qc = cirq.Circuit()

        # Initialization is skipped - computational basis is the mixed version of Hadamard basis

        for layer in range(self.num_layers):
            self.qc.append(self.cost_hamiltonian())
            self.qc.append(self.mixing_hamiltonian())
            self.qc.append(QuditBarrier(num_qudits=self.num_qudits).on(*self.qudits))

        # Measurement in H basis
        self.qc.append(cirq.Moment([QuditRGate(0, 1, sp.pi / 2, sp.pi / 2).on(q) for q in self.qudits]))
        self.qc.append(cirq.measure(*self.qudits, key='qudits'))

    def cost_hamiltonian(self, weights=None):
        """Two qudit XX rotation layer for some of pairs. Edge weight is the parameter"""
        for i in range(self.num_qudits):
            for j in range(i + 1, self.num_qudits):
                if weights is None:
                    weight = np.random.uniform(0, 20)
                else:
                    weight = weights[i][j]
                for d_i in range(4):
                    for d_j in range(d_i + 1, 4):
                        if np.random.choice([0, 1], p=[1 - self.pairing_prob, self.pairing_prob]) == 1:  # coin with pairing probability
                            mix_gate = QuditXXGate(d_i, d_j, weight)
                            yield mix_gate.on(self.qudits[i], self.qudits[j])

                            if self.with_noise:
                                yield DoubleQuquartDepolarizingChannel().on(self.qudits[i], self.qudits[j])

    def mixing_hamiltonian(self):
        """Single-qudit rotation layer"""
        for d_i in range(4):
            for d_j in range(d_i + 1, 4):
                yield cirq.Moment([QuditRGate(d_i, d_j, sp.pi, sp.pi / 2).on(q) for q in self.qudits])
        if self.with_noise:
            yield cirq.Moment([QuquartDepolarizingChannel().on(q) for q in self.qudits])


if __name__ == '__main__':

    print('QAOA Qudit Ansatz')
    vqc_ansatz = QAOAQuditAnsatz(num_qudits=2, num_layers=4, pairing_prob=0.1, with_noise=False)
    print(vqc_ansatz.qc)
    print()
