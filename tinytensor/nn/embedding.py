import numpy as np 

from tinytensor.nn.modules import Module 
from tinytensor.core.tensor import Tensor

class Embedding(Module):
    def __init__(self, num_embeddings: int, dim_embeddings:int ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim_embeddings = dim_embeddings

        weight_data = np.random.randn(num_embeddings, dim_embeddings).astype(np.float32) * 0.01
        
        self.weight = Tensor(weight_data, requires_grad=True)

    def forward(self, x):
        #x эт тензор с целыми числами
        indices = x.data.astype(np.int64)
        out_data = self.weight.data[indices]

        out = Tensor(out_data, requires_grad=self.weight.requires_grad, device=self.weight.device)

        if out.requires_grad:
            out._prev = {self.weight}

            def _backward():
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data, dtype=np.float32)
                np.add.at(self.weight.grad, indices, out.grad)

            out._backward = _backward

        return out
