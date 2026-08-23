import numpy as np
from tinytensor.core.tensor import Tensor, get_array_module
from tinytensor.nn.modules import Module
from tinytensor.nn.functional import im2col_indices, col2_im_indices

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        # если кортедж
        else:
            self.kh, self.kw = kernel_size
            
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        N, C, H, W = x.data.shape
        # меняем размерность
        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col, out_h, out_w = im2col_indices(
            x_reshaped, self.kh, self.kw, padding=self.padding, stride=self.stride
        )
        
        #находим максимумы 
        max_idx = np.argmax(x_col, axis=0)
        xpm = get_array_module(x_col)
        out = x_col[max_idx, xpm.arange(x_col.shape[1])]
        
        #сново меняем размеры
        out = out.reshape(out_h, out_w, N, C)
        out = out.transpose(2, 3, 0, 1)

        out_t = Tensor(out, requires_grad=x.requires_grad, device=x.device)

        if out_t.requires_grad:
            out_t._prev = {x}

            def _backward():
                dout = out_t.grad
                dout_flat = dout.transpose(2, 3, 0, 1).reshape(-1)

                dx_col = get_array_module(x_col).zeros_like(x_col)
                dx_col[max_idx, get_array_module(x_col).arange(max_idx.size)] = dout_flat

                dx_reshaped = col2_im_indices(
                    dx_col, x_reshaped.shape, self.kh, self.kw,
                    padding=self.padding, stride=self.stride
                )
                dx = dx_reshaped.reshape(N, C, H, W)

                if x.grad is None:
                    x.grad = get_array_module(x.data).zeros_like(x.data, dtype=np.float32)
                x.grad += dx

            out_t._backward = _backward

        return out_t


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        # если кортедж
        else:
            self.kh, self.kw = kernel_size
            
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        N, C, H, W = x.data.shape
        # меняем размерность
        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col, out_h, out_w = im2col_indices(
            x_reshaped, self.kh, self.kw, padding=self.padding, stride=self.stride
        )
        
        #находим средние
        xpa = get_array_module(x_col)
        out = xpa.mean(x_col, axis=0)
        
        #сново меняем размеры
        out = out.reshape(out_h, out_w, N, C)
        out = out.transpose(2, 3, 0, 1)

        out_t = Tensor(out, requires_grad=x.requires_grad, device=x.device)

        if out_t.requires_grad:
            out_t._prev = {x}

            def _backward():
                dout = out_t.grad
                dout_flat = dout.transpose(2, 3, 0, 1).reshape(-1)

                dx_col = get_array_module(dout_flat).repeat((dout_flat / (self.kh * self.kw))[None, :], self.kh * self.kw, axis=0)

                dx_reshaped = col2_im_indices(
                    dx_col, x_reshaped.shape, self.kh, self.kw,
                    padding=self.padding, stride=self.stride
                )
                dx = dx_reshaped.reshape(N, C, H, W)

                if x.grad is None:
                    x.grad = get_array_module(x.data).zeros_like(x.data, dtype=np.float32)
                x.grad += dx

            out_t._backward = _backward

        return out_t

class GlobalAvgPool2d(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.data.shape
        out = x.data.mean(axis=(2, 3), keepdims=True)

        out_t = Tensor(out, requires_grad=x.requires_grad, device=x.device)

        if out_t.requires_grad:
            out_t._prev = {x}

            def _backward():
                dout = out_t.grad
                dx = get_array_module(x.data).ones_like(x.data) * (dout / (H * W))
                if x.grad is None:
                    x.grad = get_array_module(x.data).zeros_like(x.data, dtype=np.float32)
                x.grad += dx
            out_t._backward = _backward

        return out_t
