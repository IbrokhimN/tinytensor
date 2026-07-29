import numpy as np
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features, 1, 1), dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_features, 1, 1), dtype=np.float32), requires_grad=True)

        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)

    def forward(self, x):
        training = self.training
        if training:
            mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x.data, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std_inv = 1.0 / np.sqrt(var + self.eps)
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
                N, C, H, W = x.data.shape
                m = N * H * W

                if self.gamma.requires_grad:
                    if self.gamma.grad is None:
                        self.gamma.grad = np.zeros_like(self.gamma.data, dtype=np.float32)
                    self.gamma.grad += np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)

                if self.beta.requires_grad:
                    if self.beta.grad is None:
                        self.beta.grad = np.zeros_like(self.beta.data, dtype=np.float32)
                    self.beta.grad += np.sum(dout, axis=(0, 2, 3), keepdims=True)

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data, dtype=np.float32)

                    if training:
                        dx_hat = dout * self.gamma.data
                        dvar = np.sum(dx_hat * x_centered * -0.5 * std_inv ** 3, axis=(0, 2, 3), keepdims=True)
                        dmean = np.sum(dx_hat * -std_inv, axis=(0, 2, 3), keepdims=True) + \
                            dvar * np.mean(-2.0 * x_centered, axis=(0, 2, 3), keepdims=True)
                        dx = dx_hat * std_inv + dvar * 2.0 * x_centered / m + dmean / m
                    else:
                        dx = dout * self.gamma.data * std_inv

                    x.grad += dx

            out._backward = _backward

        return out

