# utils

## progress_bar / train_bar

A generator wrapper that prints a progress bar over any iterable:

```python
from tinytensor.utils import train_bar   # alias for progress_bar

for epoch in train_bar(range(100), prefix="training"):
    ...
```

Prints something like:
```
training |██████████████████████████████| 100.0% [12.3s]
```

Works with anything that has `len()`, not just `range` — can also be wrapped around a `DataLoader` to show per-batch progress instead of per-epoch.

## EarlyStopping

Tracks a validation metric, stops training when it stalls, and (optionally) restores the model to its best checkpoint:

```python
from tinytensor.utils import EarlyStopping

model = MLP()
early_stopping = EarlyStopping(model, patience=7, min_delta=0.01, restore_best_weights=True)

for epoch in range(100):
    train_one_epoch(model)
    val_loss = evaluate(model)

    if early_stopping(val_loss):
        print(f"stopped at epoch {epoch+1}")
        break
```

- `patience` — number of consecutive non-improving epochs allowed before stopping
- `min_delta` — minimum improvement to count as real progress (smaller improvements don't reset the patience counter)
- `restore_best_weights` — if `True`, the model is rolled back to the weights from its best epoch on stop (via `state_dict()`/`load_state_dict()` under the hood, kept in memory through `copy.deepcopy`, nothing touches disk)

`EarlyStopping.__call__` accepts either a plain number or a `Tensor` (it unwraps `.data` itself), so the raw output of `loss_fn(...)` can be passed directly.

## summary()

```python
from tinytensor.utils import summary

model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
summary(model, input_shape=(1, 784))
```

Prints layer types, output shapes, and parameter counts:
```
==================================================
layer (type)         output shape       param #
--------------------------------------------------
Linear               (1, 128)           100,480
ReLU                  (1, 128)                0
Linear                (1, 10)             1,290
--------------------------------------------------
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
==================================================
```

Runs a dummy forward pass with zeros of the given `input_shape` to infer output shapes. Only inspects direct `Module` attributes (or lists/tuples of them) on the given model — nested containers beyond one level, or models built without storing layers as attributes, may not be fully captured.
