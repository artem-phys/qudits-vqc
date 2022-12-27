from qubit_variational_qc import *
from qudit_variational_qc import *


def ising_hamiltonian(B, n_qubits=4):

    qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    hamiltonian = 0

    for q in qubits:
        hamiltonian += -0.5 * float(B) * cirq.X(q)
    for i1 in range(len(qubits)):
        hamiltonian += -0.25 * cirq.Z(qubits[i1]) * cirq.Z(qubits[(i1 + 1) % n_qubits])

    return hamiltonian


def objective_function(params, J, B, num_qids, dimension, sv, with_noise=False):

    if dimension == 2:
        tl, pl, tqtl = qubit_parameters_reshape(params)
        vqe_circuit = qubit_variational_qc(tl, pl, tqtl, 1, num_qids, measurement=False, with_noise=with_noise)

    elif dimension == 4:
        tl, pl, tqtl = qudit_parameters_reshape(params)
        vqe_circuit = variational_qc(tl, pl, tqtl, 1, num_qids, measurement=False, with_noise=with_noise)

    simulator = cirq.Simulator()
    qubits = [cirq.LineQubit(i) for i in range(int(np.ceil(dimension ** num_qids)))]

    # Simulate
    sv = simulator.simulate(vqe_circuit, initial_state=sv).final_state_vector
    qmap = {q: i for i, q in enumerate(qubits)}
    expectation = ising_hamiltonian(B).expectation_from_state_vector(sv, qmap).real

    return expectation / 4


if __name__ == '__main__':
    print(ising_hamiltonian(0))
