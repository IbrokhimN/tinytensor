import numpy as np
import pytest

from tinytensor.core.tensor import Tensor


def test_creation_from_list_and_ndarray():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor(np.array([1, 2, 3], dtype=np.int32))
    assert t1.data.dtype == np.float32
    assert t2.data.dtype == np.float32
    assert t1.shape == (3,)


def test_add_forward_and_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    assert np.allclose(c.data, [5.0, 7.0, 9.0])
    c.backward()
    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0, 1.0])


def test_add_broadcasting_unbroadcast():
    a = Tensor(np.ones((2, 3)), requires_grad=True)
    b = Tensor(np.ones((1, 3)), requires_grad=True)
    c = a + b
    c.sum().backward()
    assert b.grad.shape == (1, 3)
    assert np.allclose(b.grad, [[2.0, 2.0, 2.0]])


def test_radd():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = 5 + a
    assert np.allclose(c.data, [6.0, 7.0])


def test_mul_forward_and_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    assert c is not None, "__mul__ должен вернуть Tensor, а не None"
    assert np.allclose(c.data, [8.0, 15.0])
    c.sum().backward()
    assert np.allclose(a.grad, [4.0, 5.0])
    assert np.allclose(b.grad, [2.0, 3.0])


def test_rmul():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = 3 * a
    assert c is not None
    assert np.allclose(c.data, [3.0, 6.0])


def test_sub_forward_and_backward():
    a = Tensor([5.0, 7.0], requires_grad=True)
    b = Tensor([1.0, 2.0], requires_grad=True)
    c = a - b
    assert np.allclose(c.data, [4.0, 5.0])
    c.backward()
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [-1.0, -1.0])


def test_rsub():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = 5 - a
    assert np.allclose(c.data, [4.0, 3.0])


def test_matmul_forward_and_backward():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), requires_grad=True)
    c = a @ b
    assert np.allclose(c.data, a.data)
    c.sum().backward()
    assert a.grad.shape == a.data.shape
    assert b.grad.shape == b.data.shape


def test_pow():
    a = Tensor([2.0, 3.0], requires_grad=True)
    c = a ** 2
    assert np.allclose(c.data, [4.0, 9.0])
    c.sum().backward()
    assert np.allclose(a.grad, [4.0, 6.0])  # d(x^2)/dx = 2x


def test_sum():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = a.sum()
    assert s.data == pytest.approx(6.0)
    s.backward()
    assert np.allclose(a.grad, [1.0, 1.0, 1.0])


def test_relu():
    a = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    out = a.relu()
    assert np.allclose(out.data, [0.0, 0.0, 2.0])
    out.sum().backward()
    assert np.allclose(a.grad, [0.0, 0.0, 1.0])


def test_leaky_relu():
    a = Tensor([-2.0, 3.0], requires_grad=True)
    out = a.leaky_relu(alpha=0.1)
    assert np.allclose(out.data, [-0.2, 3.0])
    out.sum().backward()
    assert np.allclose(a.grad, [0.1, 1.0])


def test_sigmoid_range():
    a = Tensor([-100.0, 0.0, 100.0], requires_grad=True)
    out = a.sigmoid()
    assert out.data[0] == pytest.approx(0.0, abs=1e-6)
    assert out.data[1] == pytest.approx(0.5, abs=1e-6)
    assert out.data[2] == pytest.approx(1.0, abs=1e-6)


def test_tanh():
    a = Tensor([0.0], requires_grad=True)
    out = a.tanh()
    assert out.data[0] == pytest.approx(0.0)
    out.backward()
    assert a.grad[0] == pytest.approx(1.0)  # d(tanh)/dx в 0 равен 1


def test_gelu_zero():
    a = Tensor([0.0], requires_grad=True)
    out = a.gelu()
    assert out.data[0] == pytest.approx(0.0, abs=1e-5)


def test_repr_contains_requires_grad():
    a = Tensor([1.0], requires_grad=True)
    assert "requires_grad=True" in repr(a)
