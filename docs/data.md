# tinytensor.data

Dataset loaders, the batching pipeline, and image augmentations.

```python
from tinytensor.data import load_mnist, load_fashion, DataLoader, TensorDataset
from tinytensor.data import random_flip, random_crop, Compose
```

- [Dataset loaders](#dataset-loaders) — `load_mnist`, `load_fashion`
- [Datasets](#datasets) — `Dataset`, `TensorDataset`
- [DataLoader](#dataloader)
- [Augmentations](#augmentations)

## Dataset loaders

### load_mnist

Downloads MNIST (once, cached on disk), parses the IDX format, and returns
NumPy arrays.

```python
load_mnist(data_dir="./data_mnist", normalize=True)
```

**Args:**

- `data_dir` (str): where to cache the downloaded files.
- `normalize` (bool): if `True` (default), returns images as `[N, 1, 28, 28]`
  float32 in `[0, 1]` — ready to feed to a CNN. If `False`, returns raw
  `[N, 28, 28]` uint8 in `[0, 255]`.

**Returns:** `(x_train, y_train), (x_test, y_test)`.

```python
(x_train, y_train), (x_test, y_test) = load_mnist()
```

### load_fashion

Fashion-MNIST (10 clothing classes). Same format and signature as `load_mnist`,
different source and default cache directory (`./data_fashion`).

```python
load_fashion(data_dir="./data_fashion", normalize=True)
```

## Datasets

### Dataset

Abstract base class. Subclass and implement `__len__` and `__getitem__`.

### TensorDataset

Wraps in-memory arrays `x` and `y` into a dataset.

```python
TensorDataset(x, y)
```

## DataLoader

Iterates over a dataset in batches. Wraps each batch in a `Tensor` on the
current global device (so with `set_device("cuda")`, batches are created on GPU).

```python
DataLoader(dataset, batch_size=32, shuffle=True)
```

```python
loader = DataLoader(TensorDataset(x, y), batch_size=64)
for xb, yb in loader:
    ...
```

## Augmentations

Random image transforms applied to raw NumPy batches `[N, C, H, W]` **before**
training. Use on training data only, never on the test set. They operate on the
data directly (no autograd) and never mutate the input array.

| Function | Effect |
| --- | --- |
| `random_flip(x, p=0.5)` | Horizontal mirror, per image, with probability `p`. |
| `random_crop(x, padding=4)` | Pad with zeros then crop back at a random offset (shift effect). |
| `random_rotate90(x, p=0.5)` | Rotate 90/180/270° (square images only), with probability `p`. |
| `add_noise(x, std=0.05)` | Add Gaussian noise. |
| `random_brightness(x, delta=0.2)` | Shift brightness by a random amount. |

### Compose

Chains several augmentations into one callable, applied in order.

```python
aug = Compose([random_flip, random_crop, add_noise])
x_aug = aug(x_train)
```
