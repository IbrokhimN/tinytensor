# Tensor and Autograd

## What's inside a Tensor

Every `Tensor` holds:

- `data` — the actual values (a `numpy.ndarray` on CPU, a `cupy.ndarray` on GPU, always `float32`)
- `grad` — the accumulated gradient, `None` until computed, same array type as `data`
- `_prev` — the set of parent tensors this one was built from (the computation graph edges)
- `_backward` — a closure that knows how to route the gradient from this tensor into `_prev`
- `device` — `"cpu"` or `"cuda"`

```python
from tinytensor.core.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
```

`requires_grad=False` by default — tensors that don't participate in training (input data, masks) don't need gradient bookkeeping and are skipped by the engine.

## CPU/GPU backend resolution

`tinytensor.core.tensor.get_array_module(data)` inspects the *type* of an array and returns either the `numpy` or `cupy` module:

```python
def get_array_module(data):
    if hasattr(type(data), "__module__") and "cupy" in type(data).__module__:
        import cupy as cp
        return cp
    return np
```

Every operation's backward closure calls `get_array_module(self.data)` before allocating a new gradient buffer (`zeros_like`, `ones_like`) or running an elementwise math function (`where`, `exp`, `tanh`, `sqrt`, `clip`, `maximum`, `matmul`). This means a tensor's entire forward/backward computation stays on whichever device its data actually lives on — there's no separate "GPU mode" flag to keep in sync, the correct backend is derived directly from the array in front of you at each step.

`Tensor.to(device)` / `.cuda()` / `.cpu()` move `data` (and `grad`, if already computed) between `numpy` and `cupy` arrays. `Module.to(device)` / `.cuda()` / `.cpu()` do the same recursively across every parameter and submodule, including layers stored in a `Sequential`.

## How backward() works

For `z = f(x, y)`, the chain rule gives:

```
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
```

`backward()` builds a topological order of the graph (`tinytensor/core/autograd.py`), seeds the root's gradient with ones, and walks the order in reverse, calling `_backward()` on each node. This is the same construction used in [micrograd](https://github.com/karpathy/micrograd):

```python
def backward(target_tensor):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(target_tensor)
    target_tensor.grad = np.ones_like(target_tensor.data, dtype=np.float32)

    for v in reversed(topo):
        v._backward()
```

Further reading on the general idea:
- [Karpathy — micrograd](https://github.com/karpathy/micrograd)
- [Karpathy — "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- [colah — Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)

Minimal example:

```python
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b
d = c.sum()
d.backward()
print(a.grad, b.grad)  # [3.] [2.]
```

Gradients **accumulate** rather than get overwritten. Call `optimizer.zero_grad()` (or `model.zero_grad()`) before every step, or gradients from the previous step will still be sitting on the parameters and get added to.

## Gradient formulas implemented on Tensor

| Op | Forward | Gradient |
|---|---|---|
| `a + b` | `a + b` | `dL/da += dL/dz`, `dL/db += dL/dz` (summed over broadcast axes) |
| `a * b` | `a * b` | `dL/da += dL/dz * b`, `dL/db += dL/dz * a` |
| `a @ b` | matmul (`numpy.matmul` or `cupy.matmul`, chosen via `get_array_module`) | `dL/da += dL/dz @ bᵀ`, `dL/db += aᵀ @ dL/dz` (batched: gradient uses `swapaxes(-1, -2)`, not a full transpose, so this works correctly for `(batch, heads, seq, dim)`-shaped tensors like attention scores, not just plain 2D matrices) |
| `a ** p` | `aᵖ` | `dL/da += dL/dz * p * a^(p-1)` |
| `a.reshape(...)` | reshape | `dL/da += dL/dz.reshape(a.shape)` |
| `a.transpose(*axes)` | permute axes | `dL/da += dL/dz.transpose(inverse_axes)` |
| `a.sum()` | sum | `dL/da += dL/dz * ones_like(a)` |
| ReLU | `max(0, x)` | `1` where `x>0`, else `0` |
| LeakyReLU | `x` or `αx` | `1` where `x>0`, else `α` |
| sigmoid | `1/(1+e⁻ˣ)` | `σ(x)*(1-σ(x))` |
| tanh | `tanh(x)` | `1 - tanh²(x)` |
| GELU | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` | see the [GELU paper](https://arxiv.org/abs/1606.08415), the exact formula is not short |

Layer-specific gradients (Conv2d, BatchNorm2d, RNN, Embedding, Softmax, CrossEntropyLoss) are documented in [nn.md](nn.md), since those are hand-written backward closures rather than compositions of the table above.

## Broadcasting

Why gradients sometimes need to be summed back down (`_unbroadcast`): adding a `(2,3)` tensor to a `(1,3)` tensor implicitly stretches the second one to `(2,3)` for the forward pass, so the incoming gradient on the backward pass has shape `(2,3)` and needs to be collapsed back to `(1,3)` before it matches the original tensor's shape. `_unbroadcast` only calls array *methods* (`.sum(axis=...)`, `.ndim`), which `cupy` implements identically to `numpy`, so it works correctly on either backend without needing `get_array_module` itself. See [NumPy's broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) for the underlying mechanics.

## Known limitation

`backward()` does not cache the graph between calls — every call rebuilds the topological order from scratch, same as micrograd. There is no `retain_graph`, no lazy graph construction, and in-place mutation of tensors that are part of an active graph is not tracked or protected against.
