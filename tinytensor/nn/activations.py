import numpy as np 

from tinytensor.core.tensor import Tensor 
from tinytensor.nn.modules import Module

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class LeReLU(Module):
    # у нас есть тут альфа
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x.leaky_relu(alpha=self.alpha)
    
class Sigmod(Module):
    def forward(self, x):
        return x.sigmoid()
    
class Tanh(Module):
    def forward(self, x):
        return x.tanh()
    
class GELU(Module):
    def forward(self, x):
        return x.gelu()

class Softmax(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim 

    def forward(self, x):
        max_x = np.max(x.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(x.data - max_x)

        out_data = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)

        out = Tensor(out_data, requires_grad=x.requires_grad, device=x.device)

        if out.requires_grad:
            out._prev = {x}

            def _backward():
                dy = out.grad
                dot = np.sum(dy * out_data, axis=self.dim, keepdims=True)
                dx = out_data * (dy - dot)

                if x.grad is None:
                    x.grad = np.zeros_like(x.data, dtype=np.float32)
                x.grad += dx

            out._backward = _backward

        return out



# ToDo:
# ReLU       [x]
# LeReLU     [x]
# Sigmoid    [x]
# Tanh       [x]
# GELU       [x]
# kama soska
