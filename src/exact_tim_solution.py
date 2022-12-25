import numpy as np


def exact_solution(B):

    if B == 0:
        return -0.25

    l_value = 1 / (2 * B)

    # numerical integration
    step_size = 0.0001
    k_values = np.arange(0, np.pi, step_size)
    integration_values = [step_size * np.sqrt(1 + l_value ** 2 + 2 * l_value * np.cos(ki)) for ki in k_values]
    integral = np.sum(integration_values)
    gs_energy = 1 * integral / (4 * np.pi*l_value)

    return -1 * gs_energy


if __name__ == '__main__':
    B_values = 2 ** np.linspace(0, 3, 7) - 1
    results_exact = [exact_solution(B) for B in B_values]

    print(results_exact)

