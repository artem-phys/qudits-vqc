from qudit_depolarization_channels import *
from qudit_gates import *


def rot_layer(qudits_, l1, l2, theta_array, phi_array):
    """Yields parametrized single qudit rotation"""
    for i, qid in enumerate(qudits_):
        rot = QuditRGate(l1, l2, theta_array[i], phi_array[i])
        yield rot.on(cirq.LineQid(i, dimension=4))


def rot_xx_layer(qudits_, theta):
    """Yields parametrized single qudit rotation"""
    mix_gate = QuditXXGate(0, 1, theta)
    yield mix_gate.on(*qudits_)


def one_step(qudits_, theta_lists, phi_lists, two_qudit_theta, with_noise=True):

    for l2 in 1, 2, 3:
        yield rot_layer(qudits_, 0, l2, theta_lists[l2 - 1], phi_lists[l2 - 1])

        if with_noise:
            for q in qudits_:
                yield QuquartDepolarizingChannel().on(q)

    yield rot_xx_layer(qudits_, two_qudit_theta)

    if with_noise:
        yield DoubleQuquartDepolarizingChannel().on(*qudits_)

    yield QuditBarrier().on(*qudits_)


def variational_qc(N, measurement=False, with_noise=False):

    qudits = cirq.LineQid.range(2, dimension=4)

    qc = cirq.Circuit()

    for layer in range(N):
        tl = np.random.uniform(low=0, high=2 * np.pi, size=(3, 2))
        pl = np.random.uniform(low=0, high=2 * np.pi, size=(3, 2))
        tqtl = np.random.uniform(low=0, high=2 * np.pi, size=(1, 1))[0][0]
        qc.append(one_step(qudits, tl, pl, tqtl, with_noise=with_noise))

    if measurement:
        qc.append(cirq.measure(*qudits, key='qudits'))

    return qc


if __name__ == '__main__':
    num_layers = 1  # number of variational layers

    print('Qudit Variational Ansatz')
    vqc = variational_qc(num_layers, measurement=True, with_noise=False)
    print(vqc)
    print()
