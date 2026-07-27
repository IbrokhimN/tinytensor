import numpy as np 
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module

class RNNCell(Module):
    def __init__(self,input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        std = 1.0 / np.sqrt(hidden_size)
        w_ih = np.random.uniform(-std, std, (hidden_size, input_size)).astype(np.float32)
        b_ih = np.zeros((hidden_size,), dtype=np.float32)
        
        #веса скрытог состояния
        w_hh = np.random.uniform(-std, std, (hidden_size, hidden_size)).astype(np.float32)
        b_hh = np.zeros((hidden_size,), dtype=np.float32)

        self.weight_ih = Tensor(w_ih, requires_grad=True)
        self.bias_ih = Tensor(b_ih, requires_grad=True)
        
        self.weight_hh = Tensor(w_hh, requires_grad=True)
        self.bias_hh = Tensor(b_hh, requires_grad=True)

    def forward(self, x, h=None):
        batch_size = x.data.shape[0]

        if h is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))

        # h_n=tanh(x@W_ih.T+ b_ih+h@ W_hh.T+b_hh)
        gate_x = x.data @ self.weight_ih.data.T + self.bias_ih.data
        gate_h = h.data @ self.weight_hh.data.T + self.bias_hh.data

        h_next_data = np.tanh(gate_x + gate_h)

        out = Tensor(
            h_next_data,
            requires_grad=(
                x.requires_grad or h.requires_grad
                or self.weight_ih.requires_grad or self.bias_ih.requires_grad
                or self.weight_hh.requires_grad or self.bias_hh.requires_grad
            ),
            device=x.device,
        )

        if out.requires_grad:
            out._prev = {x, h, self.weight_ih, self.bias_ih, self.weight_hh, self.bias_hh}

            def _backward():
                dgate = out.grad * (1.0 - h_next_data ** 2)

                if self.weight_ih.requires_grad:
                    if self.weight_ih.grad is None:
                        self.weight_ih.grad = np.zeros_like(self.weight_ih.data, dtype=np.float32)
                    self.weight_ih.grad += dgate.T @ x.data

                if self.bias_ih.requires_grad:
                    if self.bias_ih.grad is None:
                        self.bias_ih.grad = np.zeros_like(self.bias_ih.data, dtype=np.float32)
                    self.bias_ih.grad += np.sum(dgate, axis=0)

                if self.weight_hh.requires_grad:
                    if self.weight_hh.grad is None:
                        self.weight_hh.grad = np.zeros_like(self.weight_hh.data, dtype=np.float32)
                    self.weight_hh.grad += dgate.T @ h.data

                if self.bias_hh.requires_grad:
                    if self.bias_hh.grad is None:
                        self.bias_hh.grad = np.zeros_like(self.bias_hh.data, dtype=np.float32)
                    self.bias_hh.grad += np.sum(dgate, axis=0)

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data, dtype=np.float32)
                    x.grad += dgate @ self.weight_ih.data

                if h.requires_grad:
                    if h.grad is None:
                        h.grad = np.zeros_like(h.data, dtype=np.float32)
                    h.grad += dgate @ self.weight_hh.data

            out._backward = _backward

        return out


class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
    
    def forward(self, x, h_0=None):
        #x.shape->(batch sSize, Seq len,input size)
        batch_size, seq_len, _ = x.data.shape

        h = h_0
        outputs = []

        #по циклу по времени т
        for t in range(seq_len):
            x_t = Tensor(x.data[:, t, :], requires_grad=x.requires_grad)

            if x.requires_grad:
                x_t._prev = {x}

                def make_backward(t, x_t):
                    def _backward():
                        if x.grad is None:
                            x.grad = np.zeros_like(x.data, dtype=np.float32)
                        x.grad[:, t, :] += x_t.grad
                    return _backward

                x_t._backward = make_backward(t, x_t)

            h = self.cell(x_t, h)
            outputs.append(h)

        #сборка
        out_data = np.stack([o.data for o in outputs], axis=1)

        out = Tensor(
            out_data,
            requires_grad=any(o.requires_grad for o in outputs),
            device=x.device,
        )

        if out.requires_grad:
            out._prev = set(outputs)

            def _backward():
                for t, o in enumerate(outputs):
                    if o.requires_grad:
                        if o.grad is None:
                            o.grad = np.zeros_like(o.data, dtype=np.float32)
                        o.grad += out.grad[:, t, :]

            out._backward = _backward

        return out, h
