import numpy as np


def random_flip(x, p=0.5):
    out = x.copy()
    for i in range(len(x)):
        if np.random.rand() < p:
            out[i] = out[i][:, :, ::-1]   # переворот по ширине (последняя ось)
    return out


def random_crop(x, padding=4):
    N, C, H, W = x.shape
    # паддинг только по H и W, каналы и батч не трогаем
    padded = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    out = np.empty_like(x)
    for i in range(N):
        # случайная верхняя-левая точка выреза
        top = np.random.randint(0, 2*padding + 1)
        left = np.random.randint(0, 2*padding + 1)
        out[i] = padded[i, :, top:top+H, left:left+W]
    return out


def random_rotate90(x, p=0.5):
    out = x.copy()
    for i in range(len(x)):
        if np.random.rand() < p:
            k = np.random.randint(1, 4)   # на сколько четвертей повернуть: 1,2 или 3
            out[i] = np.rot90(out[i], k=k, axes=(1, 2))  # поворот в плоскости H,W
    return out


def add_noise(x, std=0.05):
    # добавляем гауссов шум - модель учится быть устойчивой к помехам.
    noise = np.random.randn(*x.shape).astype(x.dtype) * std
    return x + noise


def random_brightness(x, delta=0.2):
    # случайно меняем яркость: прибавляем/убавляем ко всем пикселям картинки.
    out = x.copy()
    for i in range(len(x)):
        shift = np.random.uniform(-delta, delta)
        out[i] = out[i] + shift
    return out


class Compose:
    # использование:
    #   aug = Compose([random_flip, random_crop])
    #   x_aug = aug(x_batch)
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
