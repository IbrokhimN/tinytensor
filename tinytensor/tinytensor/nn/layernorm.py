import numpy as np

from tinytensor.core.tensor import Tensor, get_array_module
from tinytensor.nn.modules import Module

# нормализация по фичам (последняя ось), а не по батчу как в BatchNorm
# идеальна для текстов/трансформеров тк работает одинаково на любом batch size
# и не зависит от статистики по батчу
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        gamma_data = np.ones(self.normalized_shape, dtype=np.float32)
        beta_data = np.zeros(self.normalized_shape, dtype=np.float32)
        self.gamma = Tensor(gamma_data, requires_grad=True)
        self.beta = Tensor(beta_data, requires_grad=True)

    def forward(self, x):
        xp = get_array_module(x.data)
        axes = tuple(range(x.data.ndim - len(self.normalized_shape), x.data.ndim))

        mean = x.data.mean(axis=axes, keepdims=True)
        var = x.data.var(axis=axes, keepdims=True)

        std_inv = 1.0 / xp.sqrt(var + self.eps)
        x_centered = x.data - mean
        x_hat = x_centered * std_inv

        out_data = self.gamma.data * x_hat + self.beta.data

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
            device=x.device,
        )

        if out.requires_grad:
            out._prev = {x, self.gamma, self.beta}

            def _backward():
                dout = out.grad
                m = 1
                for ax in axes:
                    m *= x.data.shape[ax]

                if self.gamma.requires_grad:
                    if self.gamma.grad is None:
                        self.gamma.grad = xp.zeros_like(self.gamma.data, dtype=np.float32)
                    sum_axes = tuple(i for i in range(dout.ndim) if i not in axes)
                    self.gamma.grad += xp.sum(dout * x_hat, axis=sum_axes) if sum_axes else dout * x_hat

                if self.beta.requires_grad:
                    if self.beta.grad is None:
                        self.beta.grad = xp.zeros_like(self.beta.data, dtype=np.float32)
                    sum_axes = tuple(i for i in range(dout.ndim) if i not in axes)
                    self.beta.grad += xp.sum(dout, axis=sum_axes) if sum_axes else dout

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = xp.zeros_like(x.data, dtype=np.float32)

                    dx_hat = dout * self.gamma.data
                    dvar = xp.sum(dx_hat * x_centered * -0.5 * std_inv ** 3, axis=axes, keepdims=True)
                    dmean = xp.sum(dx_hat * -std_inv, axis=axes, keepdims=True) + \
                        dvar * xp.mean(-2.0 * x_centered, axis=axes, keepdims=True)
                    dx = dx_hat * std_inv + dvar * 2.0 * x_centered / m + dmean / m

                    x.grad += dx

            out._backward = _backward

        return out
