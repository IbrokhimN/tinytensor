import numpy as np 

from tinytensor.core.tensor import Tensor, get_array_module
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
        if getattr(self, "quantized", False):
            return self._forward_quant(x)
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
                xp = get_array_module(dout)
                dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = xp.zeros_like(self.weight.data, dtype=np.float32)
                    dW = dout_reshaped @ x_col.T
                    self.weight.grad += dW.reshape(self.weight.data.shape)

                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = xp.zeros_like(self.bias.data, dtype=np.float32)
                    self.bias.grad += xp.sum(dout_reshaped, axis=1, keepdims=True)

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = xp.zeros_like(x.data, dtype=np.float32)
                    dx_col = w_row.T @ dout_reshaped
                    dx = col2_im_indices(
                        dx_col, x.data.shape, self.kh, self.kw,
                        padding=self.padding, stride=self.stride
                    )
                    x.grad += dx

            out._backward = _backward

        return out

    # ---------- int8 квантизация ----------

    def quant(self):
        # квант весов один раз, per-channel по выходным каналам
        from tinytensor.nn.linear import Linear
        w_row = np.asarray(self.weight.data, dtype=np.float32).reshape(self.out_channels, -1)
        self.w_q, self.w_scale = Linear._quant_symmetric(w_row, axis=1)
        self.b_data = None if self.bias is None else np.asarray(self.bias.data, dtype=np.float32)
        self.weight = None
        self.bias = None
        self.quantized = True
        return self

    def _forward_quant(self, x):
        from tinytensor.nn.linear import Linear
        N = x.data.shape[0]
        x_col, out_h, out_w = im2col_indices(
            np.asarray(x.data, dtype=np.float32), self.kh, self.kw,
            padding=self.padding, stride=self.stride
        )
        # вход квантуем на лету
        x_q, sx = Linear._quant_symmetric(x_col, axis=None)
        # int8 @ int8 -> int32
        acc = self.w_q.astype(np.int32) @ x_q.astype(np.int32)
        # w_scale per-channel -> нужен столбец
        y = acc.astype(np.float32) * sx * self.w_scale.reshape(-1, 1)
        if self.b_data is not None:
            y = y + self.b_data
        y = y.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
        return Tensor(y, requires_grad=False, device="cpu")

    def _quant_buffers(self):
        b = {"w_q": self.w_q, "w_scale": self.w_scale}
        if self.b_data is not None:
            b["b_data"] = self.b_data
        return b

    def _set_quant_buffers(self, sdict, prefix):
        p = (prefix + ".") if prefix else ""
        if (p + "w_q") in sdict:
            self.w_q = sdict[p + "w_q"]
            self.w_scale = sdict[p + "w_scale"]
            self.b_data = sdict.get(p + "b_data", None)
            self.weight = None
            self.bias = None
            self.quantized = True
