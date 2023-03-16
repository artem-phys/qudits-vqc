import numpy as np
import cirq

import qudit_gates, qudit_depolarization_channels

if __name__ == '__main__':
    d = 4
    length = 4

    qudits = cirq.LineQid.range(length, dimension=4)
