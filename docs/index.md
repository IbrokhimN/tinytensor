# tinytensor

Маленький самописный autograd на numpy. Тензоры, backprop, пара слоев, лоссы, оптимизаторы, даталоадер, плюс опциональный cuda-бэкенд для matmul через cublas. По сути свой мини-pytorch, только совсем без плюшек.

```bash
pip install pytinytensor
```

```python
from tinytensor.core.tensor import Tensor
from tinytensor.nn.linear import Linear
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import SGD

model = Linear(1, 1)
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)
```

Дальше по разделам слева, или сразу в [Getting Started](getting_started.md).

Исходники: [github.com/IbrokhimN/tinytensor](https://github.com/IbrokhimN/tinytensor)
