from qudit_variational_qc import *
from qubit_variational_qc import *


def qubit_parameters_reshape(params, num_qubits=4):
    tl = np.array([params[:num_qubits]])
    pl = np.array([params[num_qubits: 2 * num_qubits]])
    tqtl = np.array([params[-1]])

    return tl, pl, tqtl


def qudit_parameters_reshape(params, num_qudits=2):
    tl = [params[:num_qudits * 3].reshape((3, num_qudits))]
    pl = [params[num_qudits * 3:2 * num_qudits * 3].reshape((3, num_qudits))]
    tqtl = [params[2 * num_qudits * 3:]]

    return tl, pl, tqtl


if __name__ == '__main__':

    # Qubits d=2
    num_qubits = 4
    num_layers = 1

    params = np.random.uniform(low=0, high=2 * np.pi, size=num_qubits * 2 + 1)
    print(f'params = {params}')
    print()
    tl, pl, tqtl = qubit_parameters_reshape(params)
    print(f'theta_list = {tl}')
    print(f'phi_list = {pl}')
    print(f'tqt_list = {tqtl}')

    qubit__vqc = qubit_variational_qc(tl, pl, tqtl, num_layers, num_qubits, measurement=False, with_noise=False)
    print(qubit__vqc)

    # Qudits d=4
    num_qudits = 2
    num_layers = 1

    params = np.random.uniform(low=0, high=2 * np.pi, size=num_qudits * 6 + 1)
    print(f'params = {params}')
    print()
    tl, pl, tqtl = qudit_parameters_reshape(params)
    print(f'theta_list = {tl}')
    print(f'phi_list = {pl}')
    print(f'tqt_list = {tqtl}')
    print()
    qudit__vqc = variational_qc(tl, pl, tqtl, num_layers, num_qudits, measurement=False, with_noise=False)
    print(qudit__vqc)
