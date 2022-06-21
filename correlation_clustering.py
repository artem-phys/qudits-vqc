import sympy as sp
import numpy as np

from sympy.physics.quantum import TensorProduct, Dagger


def two_nodes_interaction(edge, d=4, num_qudits=2):
    """Two nodes interaction term for hamiltonian
        d ^ num_qudits by d ^ num_qudits matrix with eigenvalues:
         1 for same qudit states |s>
        -1 for different qudit states |s>
    """
    V_matrix = sp.zeros(d ** num_qudits, d ** num_qudits)

    u, v = edge

    for i in range(d):
        for j in range(d):
            iu_vec = sp.zeros(d, 1)
            iu_vec[i] = 1
            iv_vec = sp.zeros(d, 1)
            iu_vec[j] = 1
            if i == j:
                print(V_matrix)

import pandas as pd


def estimate_cost(graph, samples) -> float:
    """Estimate the cost function of the QAOA on the given graph using the
    provided computational basis bitstrings."""
    cost_value = 0.0

    # Loop over edge pairs and compute contribution.
    for u, v, w in graph.edges(data=True):
        u_samples = samples[str(u)]
        v_samples = samples[str(v)]

        # Determine if it was a +1 or -1 eigenvalue.
        u_signs = (-1) ** u_samples
        v_signs = (-1) ** v_samples
        term_signs = u_signs * v_signs

        # Add scaled term to total cost.
        term_val = np.mean(term_signs) * w["weight"]
        cost_value += term_val

    return -cost_value
