import numpy as np
import cirq

from qudit_gates import QuditGate, generalized_sigma, QuditGeneralizedXGate, QuditGeneralizedZGate


class QuquartDepolarizingChannel(QuditGate):

    def __init__(self, p1=None):
        super().__init__(dimension=4, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        if p1 is None:
            f1 = 0.99
            self.p1 = (1 - f1)
        else:
            self.p1 = p1

        self.mixture_probabilities = np.ones(self.d ** 2) * self.p1 / (self.d ** 2 - 1)
        self.mixture_probabilities[0] = (1 - self.p1)  # identity probability

    def _mixture_(self):

        x_unitary = QuditGeneralizedXGate(dimension=self.d).get_unitary()
        z_unitary = QuditGeneralizedZGate(dimension=self.d).get_unitary()

        ps = []
        for alpha in range(self.d):
            for beta in range(self.d):
                op = np.linalg.matrix_power(x_unitary, alpha) @ np.linalg.matrix_power(z_unitary, beta)
                ps.append(op)

        return tuple(zip(self.mixture_probabilities, ps))

    def get_mixture(self):
        return self._mixture_()

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"


class DoubleQuquartDepolarizingChannel(QuditGate):
    def __init__(self, p2=None):
        super().__init__(dimension=4, num_qubits=2)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        if p2 is None:
            f2 = 0.96
            self.p2 = (1 - f2)
        else:
            self.p2 = p2

        self.mixture_probabilities = np.ones(self.d ** 4) * self.p2 / (self.d ** 4 - 1)
        self.mixture_probabilities[0] = (1 - self.p2)  # identity probability

    def _mixture_(self):
        ps = []

        x_unitary = QuditGeneralizedXGate(dimension=self.d ** 2).get_unitary()
        z_unitary = QuditGeneralizedZGate(dimension=self.d ** 2).get_unitary()

        for alpha in range(self.d ** 2):
            for beta in range(self.d ** 2):

                op = np.linalg.matrix_power(x_unitary, alpha) @ np.linalg.matrix_power(z_unitary, beta)
                ps.append(op)

        return tuple(zip(self.mixture_probabilities, ps))

    def get_mixture(self):
        return self._mixture_()

    def _circuit_diagram_info_(self, args):
        return f"ΦΦ(p2={self.p2:.3f})", f"ΦΦ(p2={self.p2:.3f})"


if __name__ == '__main__':
    n = 2  # number of qudits
    d = 4  # dimension of qudits

    q0, q1 = cirq.LineQid.range(n, dimension=d)

    print('Ququart single depolarization channel. f1 = 0.99')
    circuit = cirq.Circuit(QuquartDepolarizingChannel().on(q0))
    print(circuit)
    print()

    print('Ququart two qudit depolarization channel. f2 = 0.96')
    circuit = cirq.Circuit(DoubleQuquartDepolarizingChannel().on(q0, q1))
    print(circuit)
    print()
