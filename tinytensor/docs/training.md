# training

There are two ways to train a model, and both work at the same time. Pick
whichever fits the task — nothing is forced on you.

- **Manual loop (torch-style)** — you write every step yourself. Full control.
- **`compile` + `fit` (keras-style)** — configure once, train in one call.

`fit()` is not a separate engine. Internally it runs the exact same
`forward → loss → zero_grad → backward → step` loop you'd write by hand — it just
lives inside a method. Both paths produce identical results.

## Manual loop

The original, fully explicit way. Use it whenever you need custom logic between
steps (gradient accumulation, custom metrics, anything non-standard).

```python
model = Sequential(Linear(16, 32), ReLU(), Linear(32, 3))
optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = MSELoss()
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

for epoch in range(5):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## compile + fit

The short way. `compile()` stores the optimizer and loss on the model; `fit()`
runs the loop.

```python
model = Sequential(Linear(16, 32), ReLU(), Linear(32, 3))
model.compile(optimizer=lambda p: AdamW(p, lr=1e-3), loss=MSELoss())
history = model.fit(x, y, epochs=5, batch_size=32)
```

### `compile(optimizer, loss)`

- `optimizer` — either a ready optimizer object, or a factory that takes params:
  - object: `model.compile(AdamW(model.parameters()), loss)`
  - factory: `model.compile(lambda p: AdamW(p, lr=1e-3), loss)` — the model
    passes its own `parameters()` in, so you don't have to.
- `loss` — any loss with a `forward(pred, target)`, i.e. `MSELoss`,
  `CrossEntropyLoss`.

Returns `self`, so you can chain: `model.compile(...).fit(...)`.

### `fit(x, y=None, epochs=1, batch_size=32, shuffle=True, verbose=True, validation_data=None, patience=None)`

Data can be passed two ways:

- **raw arrays:** `model.fit(x_np, y_np, epochs=10, batch_size=32)` — `fit` wraps
  them in a `TensorDataset` + `DataLoader` for you.
- **a ready DataLoader:** `model.fit(train_loader, epochs=10)`.

`validation_data=(x_val, y_val)` prints a `val_loss` after each epoch.

Returns a `history` dict with per-epoch losses (`history["loss"]`, and
`history["val_loss"]` if validation was given), so you can plot the training
curve afterwards.

```python
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, y_val),
)
print(history["loss"])       # [4.02, 3.46, 2.97, ...]
```

`fit()` raises if you call it before `compile()`.

### Early stopping

Pass `patience=N` to stop early when `val_loss` stops improving for `N` epochs
in a row. It's wired to the existing [`EarlyStopping`](utils.md) utility, which
also restores the best weights when it triggers. `patience=None` (the default)
means no early stopping.

```python
model.fit(x, y, epochs=100, validation_data=(x_val, y_val), patience=5)
```

Early stopping needs `validation_data` — there's nothing to monitor without it.

## evaluate

`evaluate(x, y=None, batch_size=32, verbose=True)` runs the same loop as `fit`
but *without* `zero_grad`/`backward`/`step` — it just measures the average loss
without training. Accepts data the same flexible way as `fit`.

```python
val_loss = model.evaluate(x_val, y_val)
```

`fit` uses `evaluate` internally to compute `val_loss` each epoch.
