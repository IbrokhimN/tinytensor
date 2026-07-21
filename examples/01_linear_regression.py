"""
Простая линейная регрессия: учим y = 3x + 2 на зашумленных данных.
Запуск: python examples/01_linear_regression.py

Примечание: пример использует вычитание (Tensor.__sub__) внутри MSELoss,
а __sub__ сейчас зависит от __mul__, где есть баг (нет `return out`) -
из-за этого лосс сейчас будет NaN. Баг в tinytensor/core/tensor.py не трогали.
"""
import numpy as np

from tinytensor.config import set_seed
from tinytensor.core.tensor import Tensor
from tinytensor.nn.linear import Linear
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import SGD

set_seed(42)

n_samples = 200
x_np = np.random.uniform(-5, 5, size=(n_samples, 1)).astype(np.float32)
y_np = 3 * x_np + 2 + np.random.normal(0, 0.5, size=(n_samples, 1)).astype(np.float32)

x = Tensor(x_np)
y = Tensor(y_np)

model = Linear(in_features=1, out_features=1)
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()

    pred = model(x)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | loss {float(loss.data):.4f}")

print("\nвыученные параметры:")
print("weight:", model.weight.data.flatten())
print("bias:", model.bias.data.flatten())
