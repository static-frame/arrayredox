from arrayredox import first_true_1d
import numpy as np

def test_first_true_1d():

    a = np.array([False, False, True, False])
    assert first_true_1d(a) == 2

    b = np.array([False, False, False])
    assert first_true_1d(b) == -1



