# data

## Dataset

```python
from tinytensor.data import Dataset, TensorDataset

dataset = TensorDataset(x, y)   # x, y: numpy arrays or lists of equal length
```

`TensorDataset` is a thin wrapper over a `(x, y)` pair with `__len__`/`__getitem__`. For anything custom (augmentation, on-disk loading), subclass `Dataset` and override the same two methods:

```python
class MyDataset(Dataset):
    def __len__(self):
        return ...
    def __getitem__(self, idx):
        return x_i, y_i
```

## DataLoader

```python
from tinytensor.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for xb, yb in loader:
    pred = model(xb)
    ...
```

Splits the dataset into batches, with optional shuffling (indices are reshuffled on every fresh `for ... in loader` pass, i.e. every epoch). A minimal analogue of [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html) — everything runs synchronously on the main thread, there is no multiprocessing or prefetching.

`len(loader)` is the number of batches per epoch, rounded up (`ceil`), so the last batch may be smaller than `batch_size` if the dataset size isn't a multiple of it.

`xb`/`yb` yielded by `DataLoader` are already `Tensor` instances — no manual wrapping needed. Batches are assembled on CPU via `numpy`; if your `Dataset` stores `cuda`-backed `Tensor`s, move each batch with `.cuda()` after pulling it out of the loader rather than expecting the loader itself to handle mixed backends.
