import sympy as sp
import cirq

from qudit_gates import QuditGate, generalized_sigma
from sympy.physics.quantum import TensorProduct


class QuquartDepolarizingChannel(QuditGate):

    def __init__(self, p_matrix=None):
        super().__init__(dimension=4, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.99
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)

        # Choi matrix initialization
        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * sp.ones(self.d, self.d)
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                op = TensorProduct(generalized_sigma(i, 0, 1, dimension=2), generalized_sigma(j, 0, 1, dimension=2))
                ps.append(op)
        return tuple(zip(self.p_matrix.flatten(), ps))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"


class DoubleQuquartDepolarizingChannel(QuditGate):
    def __init__(self, p_matrix=None):
        super().__init__(dimension=4, num_qubits=2)

        # Calculation of the parameter p2 based on average experimental error of two qudit gate
        f2 = 0.96
        self.p2 = (1 - f2) / (1 - 1 / (self.d ** 2) ** 2)

        # Choi matrix initialization
        self.p_matrix = self.p2 / 256 * sp.ones(16, 16) if p_matrix is None else p_matrix
        self.p_matrix[0, 0] += (1 - self.p2)  # identity probability

    def _mixture_(self):
        ps = []
        for i0 in range(self.d):
            for i1 in range(self.d):
                for i2 in range(self.d):
                    for i3 in range(self.d):
                        op = TensorProduct(TensorProduct(generalized_sigma(i0, 0, 1, dimension=2),
                                                         generalized_sigma(i1, 0, 1, dimension=2)),
                                           TensorProduct(generalized_sigma(i2, 0, 1, dimension=2),
                                                         generalized_sigma(i3, 0, 1, dimension=2)))
                        ps.append(op)
        return tuple(zip(self.p_matrix.flatten(), ps))

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
