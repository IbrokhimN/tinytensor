import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module

class Linear(Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        std = np.sqrt(2.0 / in_features)
        weightd = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.weight = Tensor(weightd, requires_grad=True)

        bias_d = np.zeros((1, out_features), dtype=np.float32)
        self.bias = Tensor(bias_d, requires_grad=True)

    def forward(self, x):
        # кароч обычная формула y = x@w + b
        return (x @ self.weight ) + self.bias