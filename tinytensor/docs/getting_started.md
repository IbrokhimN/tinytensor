# Getting Started

## Installation

```bash
pip install pytinytensor
```

From source:

```bash
git clone https://github.com/IbrokhimN/tinytensor
cd tinytensor
pip install -e .          # base install
pip install -e .[dev]     # + pytest for the test suite
```

Dependencies: `numpy` at runtime, `pybind11` at build time for the optional CUDA extension, `cupy` at runtime if you actually want tensors to live on the GPU. The build detects `nvcc` plus `libcudart`/`libcublas` automatically; if either is missing it falls back to a CPU-only build without failing the install. See [cuda.md](cuda.md) for details and troubleshooting.

## Quickstart

A minimal training loop touching the four core pieces — `Tensor`, `Linear`, `MSELoss`, `SGD`:

```python
from tinytensor.core.tensor import Tensor
from tinytensor.nn import Linear, MSELoss
from tinytensor.optim import SGD

model = Linear(in_features=1, out_features=1)
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

x = Tensor([[1.0], [2.0], [3.0]])
y = Tensor([[3.0], [5.0], [7.0]])   # y = 2x + 1

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

print(model.weight.data, model.bias.data)  # ~2.0, ~1.0
```

The loop is `zero_grad -> forward -> loss -> backward -> step`, identical in structure to PyTorch. There is no hidden machinery between those five calls — the whole autograd engine is a few hundred lines and is meant to be read, not trusted blindly (see [tensor_and_autograd.md](tensor_and_autograd.md)).

## Moving to the GPU

If `cupy` is installed and the CUDA extension was built successfully:

```python
model.cuda()          # equivalent to model.to("cuda")
x = x.cuda()
y = y.cuda()

pred = model(x)         # runs on cupy-backed data
loss = loss_fn(pred, y)
loss.backward()          # gradients accumulate as cupy arrays too
```

`model.cuda()` recursively walks every parameter and every submodule — including layers stored inside `Sequential` — so nothing is silently left on the CPU. If `cupy`/CUDA isn't available, `.cuda()` raises a clear `RuntimeError` instead of doing something silently wrong.

## A convolutional network

```python
from tinytensor.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from tinytensor.optim import AdamW

model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1),
    BatchNorm2d(8),
    ReLU(),
    MaxPool2d(2, 2),

    Conv2d(8, 16, kernel_size=3, padding=1),
    BatchNorm2d(16),
    ReLU(),
    MaxPool2d(2, 2),

    Flatten(),
    Linear(16 * 7 * 7, 64),
    ReLU(),
    Linear(64, 10),
)

loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3)
```

This exact architecture has been trained on real MNIST digits and reaches ~85% test accuracy in 5 epochs on a 4000-sample subset, on CPU, in about 20 seconds. Every layer here — `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `Flatten` — has a hand-written, gradient-checked backward pass; none of it is forward-only.

## A recurrent model

```python
import numpy as np
from tinytensor.core.tensor import Tensor
from tinytensor.nn import Module, Embedding, RNN, Linear, CrossEntropyLoss

class TinyRNNLM(Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.emb = Embedding(vocab_size, embed_dim)
        self.rnn = RNN(embed_dim, hidden_size)
        self.head = Linear(hidden_size, vocab_size)

    def forward(self, x):
        e = self.emb(x)              # (batch, seq_len, embed_dim)
        out, h = self.rnn(e)          # (batch, seq_len, hidden_size)

        # Tensor doesn't have general indexing yet, so slicing out the last
        # timestep while keeping it in the autograd graph is done by hand:
        last = Tensor(out.data[:, -1, :], requires_grad=out.requires_grad)
        if out.requires_grad:
            last._prev = {out}
            def _backward():
                if out.grad is None:
                    out.grad = np.zeros_like(out.data, dtype=np.float32)
                out.grad[:, -1, :] += last.grad
            last._backward = _backward

        return self.head(last)
```

`RNN.forward` returns a `(output, final_hidden_state)` tuple, mirroring the constructor argument order. It performs full backpropagation through time — gradients flow from the loss back through every timestep and into the `Embedding` weight table. Trained with `CrossEntropyLoss` + `AdamW` this architecture learns a random 20-token / 5-step mapping (loss 3.0 → 2.2 in 20 steps), confirming the gradient reaches all the way back to the embedding table.

## Where to go next

- [tensor_and_autograd.md](tensor_and_autograd.md) — how `Tensor` and `backward()` actually work, CPU/GPU backend resolution
- [nn.md](nn.md) — full layer reference
- [optim.md](optim.md) — `SGD`, `AdamW`, `StepLR`, gradient clipping
- [data.md](data.md) — `Dataset`/`DataLoader`
- [utils.md](utils.md) — `progress_bar`, `EarlyStopping`, `summary`
- [model_saving.md](model_saving.md) — checkpointing
- [cuda.md](cuda.md) — the CUDA backend, build and runtime
- [faq.md](faq.md) — known gotchas

Runnable example scripts live in [`examples/`](../examples/).
