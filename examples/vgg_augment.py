# ============================================================================
# VGG + аугментации (пример на синтетических данных)
# ============================================================================
# Показывает: как собрать VGG и применить аугментации к батчу перед обучением.
# Данные тут случайные (для демонстрации API); замени на load_cifar когда будет.
#
# Запуск: python vgg_augment.py
# ============================================================================

import numpy as np
from tinytensor.models import VGG
from tinytensor.optim import AdamW
from tinytensor.nn import CrossEntropyLoss
from tinytensor.data import random_flip, random_crop, Compose

# синтетические "картинки" 32x32x3, 10 классов
x = np.random.rand(64, 3, 32, 32).astype(np.float32)
y = np.random.randint(0, 10, 64).astype(np.int64)

# аугментации: склеиваем флип + сдвиг в одну цепочку
augment = Compose([random_flip, random_crop])
x_aug = augment(x)          # применяем к батчу перед обучением
print("аугментированный батч:", x_aug.shape)

# модель VGG для 32x32 цветных картинок
model = VGG(num_classes=10, in_channels=3)

model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(x_aug, y, epochs=2, batch_size=16)
