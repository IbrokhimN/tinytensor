# ============================================================================
# Пример INT8-квантизации модели в tinytensor
# ============================================================================
#
# ЧТО ТАКОЕ КВАНТИЗАЦИЯ (в двух словах)
# ------------------------------------
# Обычно веса модели хранятся как float32 - дробные числа по 4 байта каждое.
# Квантизация превращает их в int8 - целые числа от -127 до 127, по 1 байту.
# Итог: модель весит в ~4 раза меньше, а умножение матриц идёт на целых
# числах (на CPU это быстрее).
#
# Дробное число нельзя просто так засунуть в целое (0.0034 -> ?). Поэтому
# трюк такой: храним ЦЕЛОЕ приближение + один коэффициент масштаба "scale".
#   - при квантизации: целое = round(вес / scale)
#   - при использовании: вес ≈ целое * scale
# scale считается так, чтобы самый большой по модулю вес стал ровно 127.
#
# ЧТО ДЕЛАЕТ ЭТОТ ПРИМЕР
# ----------------------
#   1. Создаёт задачу классификации (можно реально проверить точность).
#   2. Обучает обычную float32-модель.
#   3. Замеряет её точность и размер весов.
#   4. Вызывает model.quant() - превращает веса в int8.
#   5. Снова замеряет точность и размер -> видно сжатие и потерю точности.
#   6. Сохраняет квантованную модель на диск и грузит обратно.
#
# Запуск:  python examples/quant_example.py
# ============================================================================

import os
import tempfile
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


# ----------------------------------------------------------------------------
# 1. ДАННЫЕ
# ----------------------------------------------------------------------------
# Делаем простую, но НЕ случайную задачу - два "облака" точек разных классов.
# Тогда модель реально чему-то учится, и падение точности после квантизации
# будет осмысленным (на чисто случайных данных точность всегда ~как повезёт).

N_PER_CLASS = 400     # точек на каждый класс
N_FEATURES = 64       # размер входного вектора
N_CLASSES = 4         # сколько классов


def make_blobs():
    xs, ys = [], []
    for c in range(N_CLASSES):
        # у каждого класса свой "центр" в пространстве признаков
        center = np.random.randn(N_FEATURES).astype(np.float32) * 3.0
        pts = center + np.random.randn(N_PER_CLASS, N_FEATURES).astype(np.float32)
        xs.append(pts)
        ys.append(np.full(N_PER_CLASS, c))
    x = np.concatenate(xs).astype(np.float32)
    labels = np.concatenate(ys)
    # перемешаем, чтобы классы не шли подряд
    idx = np.random.permutation(len(x))
    x, labels = x[idx], labels[idx]
    # one-hot: класс 2 -> [0,0,1,0]  (нужно для MSELoss)
    y_onehot = np.eye(N_CLASSES, dtype=np.float32)[labels]
    return x, y_onehot, labels


x_np, y_np, labels_np = make_blobs()

# делим на train / test (последние 200 точек - тест)
x_train, y_train = x_np[:-200], y_np[:-200]
x_test, labels_test = x_np[-200:], labels_np[-200:]

train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)


# ----------------------------------------------------------------------------
# 2. МОДЕЛЬ
# ----------------------------------------------------------------------------
# Обычный MLP: два Linear-слоя с ReLU между ними.
# Важно: квантуется только Linear. ReLU весов не имеет, его quant() пропускает.

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


model = MLP()
loss_fn = MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-3)


# ----------------------------------------------------------------------------
# 3. ОБУЧЕНИЕ (обычная float32-модель)
# ----------------------------------------------------------------------------

EPOCHS = 20
print("Обучение float32-модели...")
for epoch in range(EPOCHS):
    total = 0.0
    n = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += float(loss.data)
        n += 1
    if (epoch + 1) % 5 == 0:
        print(f"  эпоха {epoch+1:2d}/{EPOCHS}  loss={total/n:.4f}")


# ----------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: точность и размер весов
# ----------------------------------------------------------------------------

def accuracy(model, x, true_labels):
    # прогоняем тест через модель и смотрим, в скольких случаях
    # предсказанный класс (argmax) совпал с настоящим
    out = model(Tensor(x))
    pred_labels = np.argmax(np.asarray(out.data), axis=1)
    return (pred_labels == true_labels).mean()


def weights_size_bytes(model):
    # сколько байт занимают веса.
    # для float32-слоёв берём .data параметров,
    # для квантованных - int8-буферы w_q + scale + bias.
    total = 0
    for m in [model.fc1, model.fc2]:
        if getattr(m, "quantized", False):
            total += m.w_q.nbytes           # int8 веса (1 байт/число)
            total += m.w_scale.nbytes       # float32 scale (по одному на нейрон)
            if m.b_data is not None:
                total += m.b_data.nbytes    # float32 bias
        else:
            total += m.weight.data.nbytes   # float32 веса (4 байта/число)
            if m.bias is not None:
                total += m.bias.data.nbytes
    return total


# ----------------------------------------------------------------------------
# 4. ЗАМЕР ДО КВАНТИЗАЦИИ
# ----------------------------------------------------------------------------

model.eval()
acc_fp = accuracy(model, x_test, labels_test)
size_fp = weights_size_bytes(model)

print("\n--- float32 (до квантизации) ---")
print(f"  точность на тесте: {acc_fp*100:.1f}%")
print(f"  размер весов:      {size_fp} байт")


# ----------------------------------------------------------------------------
# 5. КВАНТИЗАЦИЯ - вот она, одна строчка
# ----------------------------------------------------------------------------
# model.quant() рекурсивно обходит все слои и у каждого Linear:
#   - превращает float32-веса в int8 + scale,
#   - выбрасывает старые float-веса (освобождает память),
#   - ставит флаг quantized=True, чтобы forward пошёл по целочисленному пути.

model.quant()

acc_q = accuracy(model, x_test, labels_test)
size_q = weights_size_bytes(model)

print("\n--- int8 (после квантизации) ---")
print(f"  точность на тесте: {acc_q*100:.1f}%")
print(f"  размер весов:      {size_q} байт")

print("\n--- ИТОГ ---")
print(f"  сжатие:            {size_fp/size_q:.2f}x меньше")
print(f"  потеря точности:   {(acc_fp-acc_q)*100:+.1f} процентных пункта")
# Сжатие не ровно 4x, потому что bias и scale остаются float32.
# На больших слоях их доля крошечная, и отношение стремится к 4x.


# ----------------------------------------------------------------------------
# 6. СОХРАНЕНИЕ И ЗАГРУЗКА КВАНТОВАННОЙ МОДЕЛИ
# ----------------------------------------------------------------------------
# save() кладёт в файл int8-веса + scale (а не float32), поэтому файл маленький.
# При load() новая модель сама понимает, что веса квантованы, и включает
# int8-режим - никаких дополнительных действий не нужно.

path = os.path.join(tempfile.gettempdir(), "mlp_int8.pkl")
model.save(path)
print(f"\nСохранено в {path}  ({os.path.getsize(path)} байт на диске)")

# создаём ПУСТУЮ модель и грузим в неё квантованный чекпоинт
loaded = MLP()
loaded.load(path)

acc_loaded = accuracy(loaded, x_test, labels_test)
print(f"Точность после save+load: {acc_loaded*100:.1f}%  "
      f"(совпадает с квантованной: {np.isclose(acc_loaded, acc_q)})")

os.remove(path)
print("\nГотово.")
