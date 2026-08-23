# optim

## Basic loop

`SGD` and `AdamW` are used identically:

```python
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

`optimizer.zero_grad()` and `model.zero_grad()` do the same thing (reset `.grad` on every parameter) — `Optimizer` just holds a reference to the same `parameters()` list you passed in.

## SGD

```python
from tinytensor.optim import SGD

opt = SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
```

Plain gradient descent: `w -= lr * grad`. With `momentum > 0`, a velocity buffer is kept per parameter:

```
v = momentum * v + grad
w -= lr * v
```

With `momentum=0` (default), no velocity buffers are allocated at all.

## AdamW

```python
from tinytensor.optim import AdamW

opt = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

Adam with decoupled weight decay ([Loshchilov & Hutter](https://arxiv.org/abs/1711.05101)) — unlike plain [Adam](https://arxiv.org/abs/1412.6980), the decay is applied directly to the weights rather than folded into the gradient (and therefore into the `m`/`v` moment estimates). Keeps a pair of `(m, v)` buffers per parameter plus a global step counter `t` (used for bias-correction, which matters most in the first few iterations).

Both optimizers resolve their backend per-parameter via `get_array_module(p.data)`, so they work the same way whether `parameters()` returns CPU (`numpy`) or GPU (`cupy`) tensors — no separate code path to remember.

## StepLR

```python
from tinytensor.optim import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(100):
    train_one_epoch(...)
    scheduler.step()   # multiplies optimizer.lr by gamma every step_size calls
```

Multiplies `optimizer.lr` by `gamma` every `step_size` calls to `.step()`. Call it once per epoch (not per batch) unless you specifically want per-batch decay.


## CosineAnnealingLR

Smoothly decays the learning rate following a cosine curve, from the optimizer's
starting `lr` down to `eta_min` over `T_max` epochs. The decrease is slow at the
start and end, faster in the middle — a very common schedule for CNNs and
transformers.

```python
CosineAnnealingLR(optimizer, T_max, eta_min=0.0)
```

**Args:**

- `optimizer`: the optimizer to drive.
- `T_max` (int): number of epochs to reach the minimum.
- `eta_min` (float): the floor learning rate. Default `0.0`.

Call `step()` once per epoch.

```python
scheduler = CosineAnnealingLR(optimizer, T_max=50)
for epoch in range(50):
    train_one_epoch()
    scheduler.step()
```

---

## clip_grad_norm_

```python
from tinytensor.nn.utils import clip_grad_norm_

loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Computes the global L2 norm across every parameter's **gradient** (not the parameter values), and if that norm exceeds `max_norm`, scales every gradient down in place so the resulting norm equals `max_norm`. Call it after `backward()` and before `step()`. Parameters with `grad is None` are skipped.

## Which one to reach for

For a quick check that something is learning at all, `SGD` with `momentum=0.9` is cheap and predictable. For anything larger, or when the loss is noisy, `AdamW` converges more reliably with less learning-rate tuning.
