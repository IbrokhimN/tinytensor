import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.optim import SGD, AdamW


def test_sgd_step_moves_against_gradient():
    p = Tensor([1.0, 1.0], requires_grad=True)
    p.grad = np.array([1.0, 2.0], dtype=np.float32)
    opt = SGD([p], lr=0.1)
    opt.step()
    assert np.allclose(p.data, [0.9, 0.8])


def test_sgd_zero_grad_resets_to_none():
    p = Tensor([1.0], requires_grad=True)
    p.grad = np.array([5.0], dtype=np.float32)
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None


def test_sgd_momentum_accumulates_velocity():
    p = Tensor([0.0], requires_grad=True)
    opt = SGD([p], lr=1.0, momentum=0.9)

    p.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    first = p.data.copy()

    p.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    #второй шаг должен быть больше первого изза накопленной скорости
    assert abs(p.data[0] - first[0]) > 1.0


def test_sgd_skips_params_without_grad():
    p1 = Tensor([1.0], requires_grad=True)
    p2 = Tensor([1.0], requires_grad=True)
    p1.grad = np.array([1.0], dtype=np.float32)
    p2.grad = None  # градиент не считался
    opt = SGD([p1, p2], lr=0.1)
    opt.step()
    assert p2.data[0] == 1.0  # не изменился


def test_adamw_step_changes_parameters():
    p = Tensor([1.0, -1.0], requires_grad=True)
    p.grad = np.array([0.5, -0.5], dtype=np.float32)
    opt = AdamW([p], lr=0.1)
    before = p.data.copy()
    opt.step()
    assert not np.allclose(p.data, before)
    assert opt.t == 1


def test_adamw_multiple_steps_reduce_simple_quadratic():
    p = Tensor([5.0], requires_grad=True)
    opt = AdamW([p], lr=0.5)
    for _ in range(50):
        p.grad = 2.0 * p.data
        opt.step()
    assert abs(p.data[0]) < abs(5.0)
