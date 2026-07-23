import numpy as np

from tinytensor.core.tensor import Tensor


def test_backward_sets_grad_of_ones_on_root():
    a = Tensor([1.0, 2.0], requires_grad=True)
    out = a.sum()
    out.backward()
    assert out.grad == np.float32(1.0)


def test_no_grad_when_requires_grad_false():
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([3.0, 4.0], requires_grad=False)
    c = a + b
    assert c.requires_grad is False
    assert c.grad is None  # backward не должен считаться вообще


def test_diamond_graph_accumulates_gradients():
    x = Tensor([3.0], requires_grad=True)
    y = x + x
    y.backward()
    assert np.allclose(x.grad, [2.0])


def test_chain_of_relu_and_sum():
    x = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    out = x.relu().sum()
    out.backward()
    # градиент 1 там где x>0, иначе 0
    assert np.allclose(x.grad, [0.0, 1.0, 0.0, 1.0])


def test_gradients_accumulate_across_multiple_backward_calls():
    x = Tensor([2.0], requires_grad=True)
    y = x.relu()
    y.backward()
    y2 = x.relu()
    y2.backward()
    assert np.allclose(x.grad, [2.0])


def test_backward_only_computed_once_per_node():
    calls = {"n": 0}
    x = Tensor([1.0, 2.0], requires_grad=True)
    mid = x.relu()

    original = mid._backward

    def counted():
        calls["n"] += 1
        original()

    mid._backward = counted
    out = mid.sum()
    out.backward()
    assert calls["n"] == 1
