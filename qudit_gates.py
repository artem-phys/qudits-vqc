import sympy as sp
import scipy.stats
import cirq

from sympy.physics.quantum import TensorProduct


def generalized_sigma(index, i, j, dimension=4):
    """Generalized sigma matrix for qudit gates implementation"""

    sigma = sp.zeros(dimension, dimension)

    if index == 0:
        # identity matrix elements
        sigma[i, i] = 1
        sigma[j, j] = 1
    elif index == 1:
        # sigma_x matrix elements
        sigma[i, j] = 1
        sigma[j, i] = 1
    elif index == 2:
        # sigma_y matrix elements
        sigma[i, j] = -1j
        sigma[j, i] = 1j
    elif index == 3:
        # sigma_z matrix elements
        sigma[i, i] = 1
        sigma[j, j] = -1

    return sigma


def levels_connectivity_check(l1, l2):
    """Check ion layers connectivity for gates"""
    connected_layers_list = [{0, i} for i in range(max(l1, l2) + 1)]
    assert {l1, l2} in connected_layers_list, "Layers are not connected"


class QuditGate(cirq.Gate):
    """Base class for qudits gates"""

    def __init__(self, dimension=4, num_qubits=1):
        self.d = dimension
        self.n = num_qubits
        self.symbol = None

    def _num_qubits_(self):
        return self.n

    def _qid_shape_(self):
        return (self.d,) * self.n

    def _circuit_diagram_info_(self, args):
        return (self.symbol,) * self.n


class QuditRGate(QuditGate):
    """Rotation between two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, phi, dimension=4):
        super().__init__(dimension=dimension)
        levels_connectivity_check(l1, l2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta
        self.phi = phi

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        sigma_y = generalized_sigma(2, self.l1, self.l2, dimension=self.d)

        s = sp.sin(self.phi)
        c = sp.cos(self.phi)

        u = sp.exp(-1j * self.theta / 2 * (c * sigma_x + s * sigma_y))

        return u

    def _circuit_diagram_info_(self, args):
        self.symbol = 'R'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}' + f'({self.theta}, {self.phi})'


class QuditXXGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        u = sp.exp(-1j * self.theta / 2 * TensorProduct(sigma_x, sigma_x))

        return u

    def _circuit_diagram_info_(self, args):
        self.symbol = 'XX'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({self.theta})'
        return info, info


class QuditBarrier(QuditGate):
    """Just barrier for visual separation in circuit diagrams. Does nothing"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.symbol = '|'

    def _unitary_(self):
        return sp.eye(self.d * self.d)


class QuditArbitraryUnitary(QuditGate):
    """Random unitary acts on qubits"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.unitary = sp.Matrix(scipy.stats.unitary_group.rvs(self.d ** self.n))
        self.symbol = 'U'

    def _unitary_(self):
        return self.unitary


if __name__ == '__main__':
    n = 3  # number of qudits
    d = 4  # dimension of qudits

    qudits = cirq.LineQid.range(n, dimension=d)

    alpha = sp.Symbol('alpha')
    beta = sp.Symbol('beta')

    print('Qudit R Gate')
    circuit = cirq.Circuit(QuditRGate(0, 1, alpha, beta, dimension=d).on(qudits[0]))
    print(circuit)
    print()

    print('Qudit XX Gate')
    circuit = cirq.Circuit(QuditXXGate(0, 2, beta, dimension=d).on(*qudits[:2]))
    print(circuit)
    print()

    print('Qudit Barrier')
    circuit = cirq.Circuit(QuditBarrier(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
    print()

    print('Qudit Arbitrary Unitary Gate')
    circuit = cirq.Circuit(QuditArbitraryUnitary(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
