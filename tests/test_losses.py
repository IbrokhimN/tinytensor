import numpy as np
import pytest

from tinytensor.core.tensor import Tensor
from tinytensor.nn.losses import MSELoss


def test_mse_forward_matches_manual_formula():
    y_pred = Tensor([1.0, 2.0, 3.0])
    y_true = Tensor([1.5, 2.5, 2.0])
    loss = MSELoss()(y_pred, y_true)

    expected = np.mean((y_pred.data - y_true.data) ** 2)
    assert loss.data == pytest.approx(expected, abs=1e-5)


def test_mse_zero_when_predictions_match():
    y_pred = Tensor([1.0, 2.0, 3.0])
    y_true = Tensor([1.0, 2.0, 3.0])
    loss = MSELoss()(y_pred, y_true)
    assert loss.data == pytest.approx(0.0, abs=1e-6)


def test_mse_backward_gives_gradient_wrt_prediction():
    y_pred = Tensor([2.0, 4.0], requires_grad=True)
    y_true = Tensor([0.0, 0.0])
    loss = MSELoss()(y_pred, y_true)
    loss.backward()
    # d(mean((p-t)^2))/dp = 2*(p-t)/n
    expected_grad = 2 * (y_pred.data - y_true.data) / y_pred.data.size
    assert np.allclose(y_pred.grad, expected_grad, atol=1e-5)
