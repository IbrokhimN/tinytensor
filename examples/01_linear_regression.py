"""
Showcase: тут разом собраны все текущие возможности tinytensor.
Не как отдельные примеры, а один пайплайн от и до - синтетика ->
модель -> обучение с валидацией и early stopping -> сохранение -> загрузка.

Запуск: python examples/03_showcase.py
"""
import numpy as np

from tinytensor.config import set_seed
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU
from tinytensor.nn.dropout import Dropout
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import AdamW
from tinytensor.data import TensorDataset, DataLoader
from tinytensor.utils import train_bar, EarlyStopping, summary

set_seed(0)

N_FEATURES = 20
N_CLASSES = 4
N_TRAIN = 400
N_VAL = 100


def make_fake_dataset(n):
    # синтетика вместо реального датасета, чтобы пример был самодостаточным
    x = np.random.randn(n, N_FEATURES).astype(np.float32)
    labels = np.random.randint(0, N_CLASSES, size=n)
    y_onehot = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return x, y_onehot


# 1. Tensor и autograd напрямую, без слоев - просто чтоб показать что это работает само по себе
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)
c = (a * b).sum()
c.backward()
print(f"autograd вживую: a={a.data}, b={b.data}, d(a*b).sum()/da = {a.grad}\n")


# 2. модель: Linear + ReLU + Dropout + Linear
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(N_FEATURES, 64)
        self.act1 = ReLU()
        self.drop = Dropout(p=0.3)
        self.fc2 = Linear(64, N_CLASSES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        return self.fc2(x)


model = MLP()

# 3. summary() - печатаем архитектуру перед обучением
summary(model, input_shape=(1, N_FEATURES))


# 4. данные через Dataset/DataLoader
x_train, y_train = make_fake_dataset(N_TRAIN)
x_val, y_val = make_fake_dataset(N_VAL)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

loss_fn = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)
early_stopping = EarlyStopping(model, patience=5, min_delta=1e-4)


def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()  # переключаем dropout
    total_loss, n_batches = 0.0, 0
    for xb, yb in loader:
        if train_mode:
            optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.data)
        n_batches += 1
    return total_loss / n_batches


# 5. обучение с прогрессбаром и валидацией + early stopping
EPOCHS = 30
for epoch in train_bar(range(EPOCHS), prefix="обучение"):
    train_loss = run_epoch(train_loader, train_mode=True)
    val_loss = run_epoch(val_loader, train_mode=False)

    if early_stopping(val_loss):
        print(f"\nранняя остановка на эпохе {epoch + 1} (val_loss не улучшается)")
        break

print(f"финальный train_loss={train_loss:.4f}, val_loss={val_loss:.4f}\n")


# 6. сохранение и загрузка модели
model.save("showcase_model.tt")

model2 = MLP()
model2.load("showcase_model.tt")

# проверка что веса реально совпали после save/load
same_weights = np.allclose(model.fc1.weight.data, model2.fc1.weight.data)
print(f"save/load отработал, веса совпадают: {same_weights}")
