import numpy as np 

from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
from tinytensor.nn.functional import im2col_indices, col2_im_indices


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # если как число
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        #если кортеж
        else:
            self.kh, self.kw = kernel_size

        self.stride = stride
        self.padding = padding

        n = in_channels * self.kh * self.kw
        std = np.sqrt(2.0 / n)

        w_data = np.random.randn(out_channels, in_channels, self.kh, self.kw).astype(np.float32) * std
        self.weight = Tensor(w_data, requires_grad=True)

        b_data = np.zeros((out_channels, 1), dtype=np.float32)
        self.bias = Tensor(b_data, requires_grad=True)
        
    def forward(self, x):
        N, C, H, W = x.data.shape
        
        x_col, out_h, out_w = im2col_indices(
            x.data, self.kh, self.kw, padding=self.padding, stride=self.stride
        )
        
        w_row = self.weight.data.reshape(self.out_channels, -1)

        out_data = w_row @ x_col + self.bias.data

        out_data = out_data.reshape(self.out_channels, out_h, out_w, N)
        out_data = out_data.transpose(3, 0, 1, 2)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.weight.requires_grad or self.bias.requires_grad,
            device=x.device,
        )

        if out.requires_grad:
            out._prev = {x, self.weight, self.bias}

            def _backward():
                dout = out.grad
                dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data, dtype=np.float32)
                    dW = dout_reshaped @ x_col.T
                    self.weight.grad += dW.reshape(self.weight.data.shape)

                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = np.zeros_like(self.bias.data, dtype=np.float32)
                    self.bias.grad += np.sum(dout_reshaped, axis=1, keepdims=True)

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data, dtype=np.float32)
                    dx_col = w_row.T @ dout_reshaped
                    dx = col2_im_indices(
                        dx_col, x.data.shape, self.kh, self.kw,
                        padding=self.padding, stride=self.stride
                    )
                    x.grad += dx

            out._backward = _backward

        return out
