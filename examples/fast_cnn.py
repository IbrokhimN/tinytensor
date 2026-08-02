import numpy as np
import tinytensor as tt 
from tinytensor.data import load_mnist
from tinytensor.models import ResNet
from tinytensor.optim import AdamW
from tinytensor.nn import CrossEntropyLoss

tt.set_device("cuda")
print(tt.cuda_available())
# скачивает MNIST
(x_train, y_train), (x_test, y_test) = load_mnist()

# нормализует каналы с 0-255 до 0-1
x_train = (x_train.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
x_test  = (x_test.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)

# 1 канал из за чб 10 классов и смол инпут для мелких картинок
model = ResNet(num_classes=10, in_channels=1, blocks_per_stage=(1,1,1,1), small_input=True)

#обучение
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(x_train, y_train, epochs=5, batch_size=64,
          validation_data=(x_test, y_test), patience=3)
