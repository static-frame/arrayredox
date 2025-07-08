from arrayredox import first_true_1d
import numpy as np

import pytest

#--------------------------------------------------------------------------
def test_first_true_1d_a() -> None:
    a1 = np.arange(100) == 50
    post = first_true_1d(a1, forward=True)
    assert post == 50

def test_first_true_1d_b() -> None:
    with pytest.raises(TypeError):
        a1 = [2, 4, 5,]
        first_true_1d(a1, forward=True)

def test_first_true_1d_c() -> None:
    with pytest.raises(TypeError):
        a1 = np.arange(100) == 50
        first_true_1d(a1, forward=0) # bad Boolean

def test_first_true_1d_d() -> None:
    a1 = np.arange(100) < 0
    post = first_true_1d(a1, forward=True)
    assert post == -1

def test_first_true_1d_e() -> None:
    a1 = np.arange(100)
    # only a Boolean array
    with pytest.raises(TypeError):
        post = first_true_1d(a1, forward=True)

def test_first_true_1d_f() -> None:
    a1 = (np.arange(100) == 3)[:50:2]
    assert first_true_1d(a1), 4

def test_first_true_1d_g() -> None:
    a1 = (np.arange(100) == 0).reshape(10, 10)
    with pytest.raises(TypeError):
        post = first_true_1d(a1, forward=True)

def test_first_true_1d_reverse_a() -> None:
    a1 = np.arange(100) == 50
    post = first_true_1d(a1, forward=False)
    assert post == 50

def test_first_true_1d_reverse_b() -> None:
    a1 = np.arange(100) == 0
    post = first_true_1d(a1, forward=False)
    assert post == 0

def test_first_true_1d_reverse_c() -> None:
    a1 = np.arange(100) == -1
    post = first_true_1d(a1, forward=False)
    assert post == -1

def test_first_true_1d_reverse_d() -> None:
    a1 = np.arange(100) == 99
    post = first_true_1d(a1, forward=False)
    assert post == 99

def test_first_true_1d_multi_a() -> None:
    a1 = np.isin(np.arange(100), (50, 70, 90))
    assert first_true_1d(a1, forward=True) == 50
    assert first_true_1d(a1, forward=False) == 90

def test_first_true_1d_multi_b() -> None:
    a1 = np.isin(np.arange(100), (10, 30, 50))
    assert first_true_1d(a1, forward=True) == 10
    assert first_true_1d(a1, forward=False) == 50
