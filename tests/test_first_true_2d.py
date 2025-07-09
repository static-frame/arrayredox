from socket import IP_ADD_SOURCE_MEMBERSHIP
from arrayredox import first_true_2d
import numpy as np

import pytest

#--------------------------------------------------------------------------
def test_first_true_2d_a1() -> None:
    a1 = np.arange(20).reshape(4, 5) % 3 == 0
    pos2 = first_true_2d(a1, axis=1)
    assert pos2.tolist() == [0, 1, 2, 0]

    pos1 = first_true_2d(a1, axis=0, forward=True)
    assert pos1.tolist() == [0, 1, 2, 0, 1]

    pos2 = first_true_2d(a1, axis=0, forward=False)
    assert pos2.tolist() == [3, 1, 2, 3, 1]


def test_first_true_2d_a2() -> None:
    a1 = np.isin(np.arange(100), (9, 19, 38, 68, 96)).reshape(5, 20)

    post1 = first_true_2d(a1, axis=1, forward=True)
    # NOTE: this is an axis 1 result by argmax
    assert post1.tolist() == [9, 18, -1, 8, 16]

    post2 = first_true_2d(a1, axis=1, forward=False)
    assert post2.tolist() == [19, 18, -1, 8, 16]

def test_first_true_2d_b() -> None:
    a1 = np.isin(np.arange(20), (3, 7, 10, 15, 18)).reshape(5, 4)

    post1 = first_true_2d(a1, axis=1, forward=False)
    assert post1.tolist() == [3, 3, 2, 3, 2]

    post2 = first_true_2d(a1, axis=1, forward=True)
    assert post2.tolist() == [3, 3, 2, 3, 2]

    post3 = first_true_2d(a1, axis=0, forward=False)
    assert post3.tolist() == [-1, -1, 4, 3]

    post4 = first_true_2d(a1, axis=0, forward=True)
    assert post4.tolist() == [-1, -1, 2, 0]

def test_first_true_2d_c() -> None:
    a1 = np.isin(np.arange(20), ()).reshape(5, 4)

    post1 = first_true_2d(a1, axis=1, forward=False)
    assert post1.tolist() == [-1, -1, -1, -1, -1]

    post2 = first_true_2d(a1, axis=1, forward=True)
    assert post2.tolist() == [-1, -1, -1, -1, -1]

    post3 = first_true_2d(a1, axis=0, forward=False)
    assert post3.tolist() == [-1, -1, -1, -1]

    post4 = first_true_2d(a1, axis=0, forward=True)
    assert post4.tolist() == [-1, -1, -1, -1]


def test_first_true_2d_d() -> None:
    a1 = np.isin(np.arange(20), (0, 3, 4, 7, 8, 11, 12, 15, 16, 19)).reshape(5, 4)

    post1 = first_true_2d(a1, axis=1, forward=False)
    assert post1.tolist() == [3, 3, 3, 3, 3]

    post2 = first_true_2d(a1, axis=1, forward=True)
    assert post2.tolist() == [0, 0, 0, 0, 0]

    post3 = first_true_2d(a1, axis=0, forward=True)
    assert post3.tolist() == [0, -1, -1, 0]

    post4 = first_true_2d(a1, axis=0, forward=False)
    assert post4.tolist() == [4, -1, -1, 4]

def test_first_true_2d_e() -> None:
    a1 = np.isin(np.arange(15), (2, 7, 12)).reshape(3, 5)

    post1 = first_true_2d(a1, axis=1, forward=False)
    assert post1.tolist() == [2, 2, 2]

    post2 = first_true_2d(a1, axis=1, forward=True)
    assert post2.tolist() == [2, 2, 2]

def test_first_true_2d_f() -> None:
    a1 = np.isin(np.arange(15), (2, 7, 12)).reshape(3, 5)

    with pytest.raises(ValueError):
        post1 = first_true_2d(a1, axis=-1)

    with pytest.raises(ValueError):
        post1 = first_true_2d(a1, axis=2)


def test_first_true_2d_g() -> None:
    a1 = np.isin(np.arange(15), (1, 7, 14)).reshape(3, 5)
    post1 = first_true_2d(a1, axis=0, forward=True)
    assert post1.tolist() == [-1, 0, 1, -1, 2]

    post2 = first_true_2d(a1, axis=0, forward=False)
    assert post2.tolist() == [-1, 0, 1, -1, 2]


def test_first_true_2d_h() -> None:
    a1 = np.isin(np.arange(15), (1, 7, 14)).reshape(3, 5).T # force fortran ordering
    assert first_true_2d(a1, axis=0, forward=True).tolist() == [1, 2, 4]
    assert first_true_2d(a1, axis=0, forward=False).tolist() == [1, 2, 4]
    assert first_true_2d(a1, axis=1, forward=True).tolist() == [-1, 0, 1, -1, 2]
    assert first_true_2d(a1, axis=1, forward=False).tolist() == [-1, 0, 1, -1, 2]


def test_first_true_2d_i() -> None:
    # force fortran ordering, non-contiguous, non-owned
    a1 = np.isin(np.arange(15), (1, 4, 5, 7, 8, 12, 15)).reshape(3, 5).T[:4]
    assert first_true_2d(a1, axis=0, forward=True).tolist() == [1, 0, 2]
    assert first_true_2d(a1, axis=0, forward=False).tolist() == [1, 3, 2]
    assert first_true_2d(a1, axis=1, forward=True).tolist() == [1, 0, 1, 1]
    assert first_true_2d(a1, axis=1, forward=False).tolist() == [1, 0, 2, 1]

