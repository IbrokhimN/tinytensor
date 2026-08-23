# ============================================================================
# Демонстрация возможностей tinytensor - всё что добавили
# ============================================================================
# Прогоняет по очереди: датасеты, аугментации, лоссы, архитектуры,
# scheduler, обучение с accuracy. Запусти и смотри что выводится.
#
# Запуск: python showcase.py
# Для GPU: раскомментируй две строки с set_device ниже.
# ============================================================================

import numpy as np
import tinytensor as tt
tt.set_device('cuda')   # <- раскомментируй для обучения на видюхе

from tinytensor.data import load_mnist, load_fashion
from tinytensor.data import random_flip, random_crop, add_noise, Compose
from tinytensor.models import LeNet, VGG, ResNet18
from tinytensor.optim import AdamW, CosineAnnealingLR
from tinytensor.nn import CrossEntropyLoss, BCELoss, MSELoss
from tinytensor.core.tensor import Tensor


def line(title):
    print("\n" + "=" * 55)
    print("  " + title)
    print("=" * 55)


# ---------------------------------------------------------------------------
line("1. ДАТАСЕТЫ - загрузка одной строкой")
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = load_mnist()
print(f"MNIST train: {x_train.shape}, метки {y_train.shape}")
print(f"диапазон пикселей: {x_train.min():.1f} - {x_train.max():.1f} (уже нормализовано)")
# fashion качается так же: load_fashion()


# ---------------------------------------------------------------------------
line("2. АУГМЕНТАЦИИ - случайные искажения картинок")
# ---------------------------------------------------------------------------
batch = x_train[:16]
print(f"исходный батч: {batch.shape}")

flipped = random_flip(batch, p=1.0)
print(f"после flip:    {flipped.shape}  (отзеркалено)")

# цепочка аугментаций
augment = Compose([random_flip, random_crop, add_noise])
augmented = augment(batch)
print(f"после Compose (flip+crop+noise): {augmented.shape}")


# ---------------------------------------------------------------------------
line("3. НОВЫЕ ОПЕРАЦИИ ТЕНЗОРА - abs и log")
# ---------------------------------------------------------------------------
t = Tensor(np.array([-2.0, 3.0, -1.0]), requires_grad=True)
print(f"abs([-2, 3, -1]) = {t.abs().data.tolist()}")

t2 = Tensor(np.array([1.0, np.e]), requires_grad=True)
print(f"log([1, e])      = {[round(v, 2) for v in t2.log().data.tolist()]}")


# ---------------------------------------------------------------------------
line("4. ЛОССЫ - MSE, CrossEntropy, BCE")
# ---------------------------------------------------------------------------
# BCE для бинарной классификации
p = Tensor(np.array([[0.9], [0.1], [0.8]]))   # предсказанные вероятности
y = Tensor(np.array([[1.0], [0.0], [1.0]]))   # настоящие метки 0/1
print(f"BCELoss (хорошее предсказание): {float(BCELoss()(p, y).data):.4f}")

p_bad = Tensor(np.array([[0.1], [0.9], [0.2]]))
print(f"BCELoss (плохое предсказание):  {float(BCELoss()(p_bad, y).data):.4f}")


# ---------------------------------------------------------------------------
line("5. SCHEDULER - CosineAnnealing плавно снижает lr")
# ---------------------------------------------------------------------------
tmp = LeNet(num_classes=10, in_channels=1)
opt = AdamW(tmp.parameters(), lr=0.1)
sched = CosineAnnealingLR(opt, T_max=5)
lrs = [round(opt.lr, 4)]
for _ in range(5):
    sched.step()
    lrs.append(round(opt.lr, 4))
print(f"lr по эпохам: {lrs}")


# ---------------------------------------------------------------------------
line("6. АРХИТЕКТУРЫ - размеры и параметры")
# ---------------------------------------------------------------------------
for name, model in [("LeNet", LeNet(10, 1)), ("VGG", VGG(10, 3)), ("ResNet18", ResNet18(10, 1))]:
    n = sum(pp.data.size for pp in model.parameters())
    print(f"{name:9s}: {n:>12,} параметров")


# ---------------------------------------------------------------------------
line("7. ОБУЧЕНИЕ - LeNet на MNIST (с accuracy и прогресс-баром)")
# ---------------------------------------------------------------------------
model = LeNet(num_classes=10, in_channels=1)
model.compile(lambda pp: AdamW(pp, lr=1e-3), CrossEntropyLoss())

# берём подвыборку чтоб демо было быстрым
print("обучаю на 2000 примерах, 3 эпохи:\n")
history = model.fit(
    x_train[:2000], y_train[:2000],
    epochs=3,
    batch_size=64,
    validation_data=(x_test[:500], y_test[:500]),
)

print(f"\nитоговая точность на train: {history['acc'][-1]*100:.1f}%")
print("\nГОТОВО - все фичи работают!")
