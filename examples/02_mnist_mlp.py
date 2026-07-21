#MLP-классификатор на синтетических данных в стиле MNIST
#(784 признака = 28x28, 10 классов). Реального MNIST тут нет -
#в tinytensor нет загрузчика датасетов и torchvision, поэтому
#генерируем случайные "картинки" и метки, чтобы показать пайплайн
#Dataset -> DataLoader -> MLP (Linear+ReLU) -> MSE на one-hot -> AdamW.
#
#Запуск python examples/02_mnist_mlp.py

import numpy as np

from tinytensor.config import set_seed
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import AdamW
from tinytensor.data import TensorDataset, DataLoader

set_seed(0)

N_SAMPLES = 512
N_FEATURES = 784  # 28*28
N_CLASSES = 10


def make_fake_mnist(n):
    x = np.random.randn(n, N_FEATURES).astype(np.float32)
    labels = np.random.randint(0, N_CLASSES, size=n)
    y_onehot = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return x, y_onehot


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(N_FEATURES, 128)
        self.act = ReLU()
        self.fc2 = Linear(128, N_CLASSES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


x_np, y_np = make_fake_mnist(N_SAMPLES)
dataset = TensorDataset(x_np, y_np)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MLP()
loss_fn = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        optimizer.zero_grad()

        pred = model(xb)
        loss = loss_fn(pred, yb)

        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.data)
        n_batches += 1

    print(f"epoch {epoch + 1}/{epochs} | avg loss {epoch_loss / n_batches:.4f}")
