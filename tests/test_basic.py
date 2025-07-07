from arrayredox import first_true_1d
import numpy as np

def test_first_true_1d():

    a = np.array([False, False, True, False])
    assert first_true_1d(a) == 2

    b = np.array([False, False, False])
    assert first_true_1d(b) == -1


#--------------------------------------------------------------------------
def test_first_true_1d_a() -> None:
    a1 = np.arange(100) == 50
    post = first_true_1d(a1, forward=True)
    assert post == 50

# def test_first_true_1d_b() -> None:
#     with self.assertRaises(TypeError):
#         a1 = [2, 4, 5,]
#         first_true_1d(a1, forward=True)

# def test_first_true_1d_c() -> None:
#     with self.assertRaises(ValueError):
#         a1 = np.arange(100) == 50
#         first_true_1d(a1, forward=a1)

def test_first_true_1d_d() -> None:
    a1 = np.arange(100) < 0
    post = first_true_1d(a1, forward=True)
    assert post == -1

# def test_first_true_1d_e() -> None:
#     a1 = np.arange(100)
#     # only a Boolean array
#     with self.assertRaises(ValueError):
#         post = first_true_1d(a1, forward=True)

# def test_first_true_1d_f() -> None:
#     a1 = (np.arange(100) == 0)[:50:2]
#     # only a contiguous array
#     with self.assertRaises(ValueError):
#         post = first_true_1d(a1, forward=True)

# def test_first_true_1d_g() -> None:
#     a1 = (np.arange(100) == 0).reshape(10, 10)
#     # only a contiguous array
#     with self.assertRaises(ValueError):
#         post = first_true_1d(a1, forward=True)

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
