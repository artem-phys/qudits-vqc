import numpy as np

from qudit_depolarization_channels import *
from qudit_gates import *


def rot_layer(qudits_, l1, l2, theta_array, phi_array, with_noise=False):
    """Yields parametrized single qudit rotations"""
    for i, qid in enumerate(qudits_):
        rot = QuditRGate(l1, l2, theta_array[i], phi_array[i])
        yield rot.on(cirq.LineQid(i, dimension=4))
        if with_noise:
            yield QuquartDepolarizingChannel().on(qid)


def rot_zz_layer(qudits_, theta, with_noise=False):
    """Yields parametrized two qudit rotation"""
    mix_gate = QuditZZGate(0, 1, theta)
    for i in range(len(qudits_)):
        for j in range(i + 1, len(qudits_)):
            yield mix_gate.on(qudits_[i], qudits_[j])
            if with_noise:
                yield DoubleQuquartDepolarizingChannel().on(qudits_[i], qudits_[j])


def one_step(qudits_, theta_lists, phi_lists, two_qudit_theta, with_noise=False):
    """One variational step"""
    for l2 in 1, 2, 3:
        yield rot_layer(qudits_, 0, l2, theta_lists[l2 - 1], phi_lists[l2 - 1], with_noise=with_noise)

    yield rot_zz_layer(qudits_, two_qudit_theta[0], with_noise=with_noise)

    yield QuditBarrier(num_qudits=len(qudits_)).on(*qudits_)


def variational_qc(tl_, pl_, tqtl_, num_layers_, num_qudits_, measurement=False, with_noise=False):

    # two_qudits
    qudits_ = cirq.LineQid.range(num_qudits_, dimension=4)
    qc = cirq.Circuit()

    for layer in range(num_layers_):
        qc.append(one_step(qudits_, tl_[layer], pl_[layer], tqtl_[layer], with_noise=with_noise))

    if measurement:
        qc.append(cirq.measure(*qudits_, key='qudits'))

    return qc


if __name__ == '__main__':
    num_layers = 1  # number of variational layers
    num_qudits = 4
    num_qudits_pairs = int((num_qudits + 1) * num_qudits / 2)

    tl = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, 3, num_qudits))
    pl = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, 3, num_qudits))
    tqtl = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, num_qudits_pairs))

    print('Qudit Variational Ansatz')
    vqc = variational_qc(tl, pl, tqtl, num_layers, num_qudits, measurement=True, with_noise=False)
    print(vqc)
    print()
