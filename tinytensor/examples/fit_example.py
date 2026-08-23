# ============================================================================
# Гибкое обучение: два стиля на выбор
# ============================================================================
#
# tinytensor не заставляет обучать модель каким-то одним способом.
# Есть ДВА пути, и оба работают одновременно:
#
#   СТИЛЬ TORCH (ручной цикл)   - полный контроль, пишешь каждый шаг сам.
#   СТИЛЬ KERAS (model.fit)     - коротко: compile один раз, потом fit.
#
# Внутри fit() делает ровно тот же цикл, что ты писал бы руками -
# никакой магии, просто удобная обёртка. Выбирай что нравится.
#
# Запуск:  python examples/fit_example.py
# ============================================================================

import numpy as np
from tinytensor.config import set_seed
from tinytensor.nn.modules import Sequential
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import AdamW
from tinytensor.data import TensorDataset, DataLoader

# общие данные для обоих примеров
x = np.random.randn(300, 16).astype(np.float32)
y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 300)]
x_train, y_train = x[:250], y[:250]
x_val, y_val = x[250:], y[250:]


# ----------------------------------------------------------------------------
# СТИЛЬ KERAS - compile + fit
# ----------------------------------------------------------------------------
# Самый короткий путь. Оптимайзер можно передать фабрикой (lambda p: ...),
# тогда не надо самому доставать model.parameters().
print("=== KERAS-STYLE (model.fit) ===")
set_seed(0)

model = Sequential(Linear(16, 32), ReLU(), Linear(32, 3))
model.compile(
    optimizer=lambda p: AdamW(p, lr=1e-3),   # фабрика: параметры подставятся сами
    loss=MSELoss(),
)

# fit принимает сырые массивы и сам собирает батчи.
# validation_data - опционально, печатает val_loss после каждой эпохи.
# Возвращает history (лоссы по эпохам) - как в Keras.
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_val, y_val),
)
print("history['loss']:", [round(v, 4) for v in history["loss"]])

# отдельно посчитать лосс на любых данных без обучения:
model.evaluate(x_val, y_val)


# ----------------------------------------------------------------------------
# СТИЛЬ TORCH - ручной цикл
# ----------------------------------------------------------------------------
# Ровно то же самое, но каждый шаг виден и под контролем.
# Нужно, когда хочешь что-то нестандартное: свою логику между шагами,
# накопление градиентов, кастомные метрики и т.п.
print("\n=== TORCH-STYLE (ручной цикл) ===")
set_seed(0)

model2 = Sequential(Linear(16, 32), ReLU(), Linear(32, 3))
optimizer = AdamW(model2.parameters(), lr=1e-3)
loss_fn = MSELoss()
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

for epoch in range(5):
    total, n = 0.0, 0
    for xb, yb in loader:
        pred = model2(xb)          # forward
        loss = loss_fn(pred, yb)   # ошибка

        optimizer.zero_grad()      # обнулить градиенты
        loss.backward()            # backprop
        optimizer.step()           # шаг

        total += float(loss.data)
        n += 1
    print(f"эпоха {epoch+1}/5  loss={total/n:.4f}")

# ----------------------------------------------------------------------------
# Оба варианта дают одинаковый результат - fit() это просто этот цикл,
# завёрнутый в метод. Пользуйся тем, что удобнее под задачу.
# ----------------------------------------------------------------------------
