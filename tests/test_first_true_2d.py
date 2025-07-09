from arrayredox import first_true_2d
import numpy as np

import pytest

#--------------------------------------------------------------------------
def test_first_true_2d_a() -> None:
    a1 = np.arange(20).reshape(4, 5) % 3 == 0
    pos2 = first_true_2d(a1, 1)
    assert pos2.tolist() == [0, 1, 2, 0]

    pos1 = first_true_2d(a1, 0)
    import ipdb; ipdb.set_trace()
    assert pos1.tolist() == [0, 1, 2, 0, 1]




