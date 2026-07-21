import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU, LeReLU, Sigmod, Tanh, GELU


def test_module_parameters_collects_only_requires_grad_tensors():
    class Dummy(Module):
        def __init__(self):
            super().__init__()
            self.w = Tensor([1.0], requires_grad=True)
            self.const = Tensor([2.0], requires_grad=False)

    m = Dummy()
    params = m.parameters()
    assert len(params) == 1
    assert params[0] is m.w


def test_module_parameters_recurse_into_submodules():
    class Inner(Module):
        def __init__(self):
            super().__init__()
            self.w = Tensor([1.0], requires_grad=True)

    class Outer(Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.b = Tensor([2.0], requires_grad=True)

    m = Outer()
    assert len(m.parameters()) == 2


def test_module_zero_grad():
    class Dummy(Module):
        def __init__(self):
            super().__init__()
            self.w = Tensor([1.0], requires_grad=True)

    m = Dummy()
    m.w.grad = np.array([9.0], dtype=np.float32)
    m.zero_grad()
    assert m.w.grad is None


def test_linear_forward_shape():
    layer = Linear(in_features=4, out_features=3)
    x = Tensor(np.random.randn(5, 4))
    out = layer(x)
    assert out.shape == (5, 3)


def test_linear_has_weight_and_bias_as_params():
    layer = Linear(3, 2)
    params = layer.parameters()
    assert len(params) == 2  # weight + bias
    assert layer.weight.shape == (3, 2)
    assert layer.bias.shape == (1, 2)


def test_relu_module_matches_tensor_method():
    x = Tensor([-1.0, 2.0])
    assert np.allclose(ReLU()(x).data, x.relu().data)


def test_leaky_relu_module():
    x = Tensor([-2.0, 3.0])
    out = LeReLU(alpha=0.2)(x)
    assert np.allclose(out.data, [-0.4, 3.0])


def test_sigmoid_module():
    x = Tensor([0.0])
    out = Sigmod()(x)
    assert np.allclose(out.data, [0.5])


def test_tanh_module():
    x = Tensor([0.0])
    out = Tanh()(x)
    assert np.allclose(out.data, [0.0])


def test_gelu_module():
    x = Tensor([0.0])
    out = GELU()(x)
    assert np.allclose(out.data, [0.0], atol=1e-5)
