# data

## Dataset

```python
from tinytensor.data import Dataset, TensorDataset

dataset = TensorDataset(x, y)   # x, y - numpy-массивы или списки одинаковой длины
```

`TensorDataset` это просто обертка над парой `(x, y)` с `__len__`/`__getitem__`. Если нужен свой датасет (например с аугментациями или чтением с диска) - наследуйтесь от `Dataset` и переопределяйте те же два метода:

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

Бьет датасет на батчи, можно с shuffle (перемешивает индексы каждую новую эпоху, то есть на каждый `for ... in loader` заново). Мини-версия [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html), никакого multiprocessing/prefetch тут нет - все синхронно в основном потоке.

`len(loader)` возвращает количество батчей за эпоху, округленное вверх (`ceil`), то есть последний батч может быть меньше `batch_size`, если размер датасета на него не делится ровно.

`xb`/`yb` которые отдает `DataLoader` - уже готовые `Tensor`, руками оборачивать не надо.
