from qudit_gates import *


def fixed_qubit_rot_layer(qubits_, theta, phi, with_noise=False):
    """Yields parametrized single qubit rotations"""
    for i, qid in enumerate(qubits_):
        rot = QuditRGate(0, 1, theta, phi, dimension=2)
        yield rot.on(cirq.LineQid(i, dimension=2))
        if with_noise:
            p1 = 0.001  # single qubit
            yield cirq.DepolarizingChannel(p=p1, n_qubits=1).on(qid)


def qubit_rot_zz_layer(qubits_, theta, with_noise=False):
    """Yields parametrized two qudit rotation"""
    for i in range(len(qubits_)):
        for j in range(i + 1, len(qubits_)):
            yield cirq.ZZ(qubits_[i], qubits_[j]) ** theta
            if with_noise:
                p2 = 0.01  # two qubit
                yield cirq.DepolarizingChannel(p=p2, n_qubits=2).on(qubits_[i], qubits_[j])


def fixed_qubit_one_step(qubits_, theta, phi, two_qudit_theta, with_noise=False):
    """One variational step"""
    yield fixed_qubit_rot_layer(qubits_, theta, phi, with_noise=with_noise)

    yield qubit_rot_zz_layer(qubits_, two_qudit_theta, with_noise=with_noise)


def fixed_qubit_variational_qc(tl, pl, tqtl, num_layers_, num_qubits_, measurement=False, with_noise=False):

    # two_qudits
    qubits_ = cirq.LineQid.range(num_qubits_, dimension=2)
    qc = cirq.Circuit()

    # qc.append([cirq.H(q) for q in qubits_], strategy=InsertStrategy.NEW_THEN_INLINE)

    for layer in range(num_layers_):
        qc.append(fixed_qubit_one_step(qubits_, tl[layer], pl[layer], tqtl[layer], with_noise=with_noise), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    if measurement:
        qc.append(cirq.measure(*qubits_, key='qubits'))

    return qc


def qubit_parameters_reshape(params, num_qubits=4):
    tl = np.array([params[:num_qubits]])
    pl = np.array([params[num_qubits: 2 * num_qubits]])
    tqtl = np.array([params[-1]])

    return tl, pl, tqtl


if __name__ == '__main__':
    num_layers = 1
    num_qubits = 4
    num_qubits_pairs = int((num_qubits + 1) * num_qubits / 2)

    tl_ = np.random.uniform(low=0, high=2 * np.pi, size=num_layers)
    pl_ = np.random.uniform(low=0, high=2 * np.pi, size=num_layers)
    tqtl_ = np.random.uniform(low=0, high=2 * np.pi, size=num_layers)

    print('Qudit Variational Ansatz')
    qubit_vqc = fixed_qubit_variational_qc(tl_, pl_, tqtl_, num_layers, num_qubits, measurement=True, with_noise=False)
    print(qubit_vqc)
    print()
