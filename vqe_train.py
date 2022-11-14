import cirq
import numpy as np


def train(b_field, max_layers, n_qubits, dimension, n_initial, verbose=False, with_noise=False):

    layers_energy = []
    initial_statevector = np.zeros(16, dtype='complex')
    initial_statevector[0] += 1
    sv = initial_statevector

    for layer in range(1, max_layers + 1):

        cost_energy = []
        angles = []

        if verbose:
            print(f'B = {b_field}, layer = {layer}')

        for ii in range(n_initial):
            #print counter

            # randomly initialize variational parameters within appropriate bounds
            np.random.seed(ii)
            if dimension == 2:
                params_size = num_qubits * 2 + 1
            elif dimension == 4:
                params_size = num_qudits * 6 + 1

            #initial params
            if ii == 0:
                params0 = np.zeros(params_size).tolist()
            else:
                params0 = np.random.uniform(low=0, high=2 * np.pi, size=params_size).tolist()

            bnds = [(0, 2 * np.pi) for _ in range(int(len(params0)))]

            # run classical optimization
            if dimension == 2:
                result = minimize(objective_function, params0, args=(J, b_field, num_qubits, dimension, sv),
                            method='Nelder-Mead', bounds=bnds, options={'maxiter': 100})
            elif dimension == 4:
                result = minimize(objective_function, params0, args=(J, b_field, num_qudits, dimension, sv),
                            method='Nelder-Mead', bounds=bnds, options={'maxiter': 100})


            # store result of classical optimization
            result_energy = result.fun
            cost_energy.append(result_energy)
            result_angle = result.x
            angles.append(result_angle)

        # store energy minimum (over different initial configurations)
        energy_min = np.min(cost_energy)
        optim_angles = angles[np.argmin(cost_energy)]
        if verbose:
            print()
            print('Energy per initial seeds:', cost_energy)
            print('Best energy:', energy_min)
            print('Best angles:', optim_angles)
            print()

        layers_energy.append(energy_min)

        # Parameters reshape
        if dimension == 2:
            tl, pl, tqtl = qubit_parameters_reshape(optim_angles)
            vqe_circuit = qubit_variational_qc(tl, pl, tqtl, 1, num_qubits, measurement=False, with_noise=with_noise)
        elif dimension == 4:
            tl, pl, tqtl = qudit_parameters_reshape(optim_angles)
            vqe_circuit = variational_qc(tl, pl, tqtl, 1, num_qudits, measurement=False, with_noise=with_noise)

        # VQE circuit simulation-
        sv = simulator.simulate(vqe_circuit, initial_state=sv).final_state_vector

    return layers_energy
