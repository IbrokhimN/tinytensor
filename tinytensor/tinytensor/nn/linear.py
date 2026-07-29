import numpy as np

from tinytensor.core.tensor import Tensor, HAS_CUDA, cp
from tinytensor.nn.modules import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = str(device).lower()

        std = np.sqrt(2.0 / in_features)

        if HAS_CUDA and self.device == "cuda":
            w_data = (cp.random.randn(in_features, out_features) * std).astype(cp.float32)
            b_data = cp.zeros((1, out_features), dtype=cp.float32) if bias else None
        else:
            w_data = (np.random.randn(in_features, out_features) * std).astype(np.float32)
            b_data = np.zeros((1, out_features), dtype=np.float32) if bias else None

        self.weight = Tensor(w_data, requires_grad=True, device=self.device)
        self.bias = Tensor(b_data, requires_grad=True, device=self.device) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
