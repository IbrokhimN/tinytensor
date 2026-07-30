# tinytensor

A small, dependency-light autograd engine and neural network toolkit built on top of NumPy. It implements a reverse-mode automatic differentiation engine (the same core idea as [micrograd](https://github.com/karpathy/micrograd)), a set of standard layers (Linear, Conv2d, RNN, Embedding, BatchNorm2d, Dropout...), optimizers, a data pipeline, and an optional CUDA backend via `cupy` for real GPU-resident tensors.

It is not a PyTorch replacement in terms of performance or ecosystem. It exists to be small enough to read end to end in an evening, while still being capable enough to train real convolutional and recurrent models on real data (MNIST, small RNN language models).

## Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Quickstart](#quickstart)
- [What's implemented](#whats-implemented)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [References](#references)

## Installation

```bash
pip install pytinytensor
```

From source:

```bash
git clone https://github.com/IbrokhimN/tinytensor
cd tinytensor
pip install -e .          # base install
pip install -e .[dev]     # + pytest for running the test suite
```

The only hard dependency is `numpy`. `pybind11` is required at build time for the optional CUDA extension, and `cupy` is required at *runtime* for actual GPU-resident tensors (`device="cuda"`). If `nvcc` plus `libcudart`/`libcublas` are found at install time, the CUDA extension is compiled; if `cupy` is also importable at runtime, `HAS_CUDA` is `True` and `.to("cuda")`/`.cuda()` move tensor data onto the GPU for real. If either piece is missing, everything silently falls back to a pure-NumPy CPU path — the install never fails because of a missing CUDA toolchain. See [docs/cuda.md](docs/cuda.md).

## Documentation

- [docs/getting_started.md](docs/getting_started.md) — installation, quickstart, first training loop
- [docs/tensor_and_autograd.md](docs/tensor_and_autograd.md) — how `Tensor` and `backward()` work, gradient formulas, CPU/GPU backend resolution
- [docs/nn.md](docs/nn.md) — every layer: `Linear`, `Conv2d`, `MaxPool2d`/`AvgPool2d`, `BatchNorm2d`, `Flatten`, `Embedding`, `RNNCell`/`RNN`, activations, `Dropout`, `Sequential`, loss functions
- [docs/training.md](docs/training.md) — the two training styles: the manual loop and keras-style `compile` / `fit` / `evaluate` (with early stopping)
- [docs/quantization.md](docs/quantization.md) — post-training INT8 quantization: `model.quant()`, the math, saving/loading quantized models
- [docs/optim.md](docs/optim.md) — `SGD`, `AdamW`, `StepLR`, `clip_grad_norm_`
- [docs/data.md](docs/data.md) — `Dataset`, `TensorDataset`, `DataLoader`
- [docs/utils.md](docs/utils.md) — `progress_bar`, `EarlyStopping`, `summary`
- [docs/model_saving.md](docs/model_saving.md) — `save`/`load`, `.tt` format
- [docs/cuda.md](docs/cuda.md) — the CUDA backend (build + runtime), `get_array_module`, troubleshooting
- [docs/faq.md](docs/faq.md) — issues that have already come up during development, kept here so they don't come up twice

## Quickstart

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

Moving a whole model to the GPU (if `cupy` + the CUDA extension are available):

```python
model.cuda()               # or model.to("cuda")
x = x.cuda()
pred = model(x)             # runs on cupy-backed tensors, gradients too
```

A CNN trained on real MNIST digits, and a small RNN language model trained with `Embedding` + `RNN` + `CrossEntropyLoss`, are both fully working end to end — see [docs/nn.md](docs/nn.md) for architecture examples.

## What's implemented

**Core**
- `Tensor` with reverse-mode autograd: `+ - * @ **`, `.sum()`, `.reshape()`, `.transpose()`, broadcasting-aware gradients, correct gradients through batched (4D+) matrix multiplication
- Backend resolved per-tensor via `get_array_module()` — every op (not just matmul) dispatches to `numpy` or `cupy` depending on where the data actually lives, so gradients stay on the correct device throughout a training step
- Activations with gradients: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`

**Layers (`tinytensor.nn`)**
- `Linear`, `Sequential`
- `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `BatchNorm2d` — full backward via im2col/col2im, gradient-checked
- `LayerNorm` — feature-axis normalization, used by the transformer block below
- `MultiHeadAttention` — scaled dot-product self-attention with optional causal masking, gradient-checked through a full multi-head block
- `Embedding` — lookup table with scatter-add gradient
- `RNNCell`, `RNN` — full backpropagation through time (BPTT)
- `ReLU`, `LeReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`
- `Dropout` — inverted dropout, gated by `model.train()`/`model.eval()`, backend-aware mask generation
- `MSELoss`, `CrossEntropyLoss`
- `Module.to(device)` / `.cuda()` / `.cpu()` — recursively move every parameter (and every submodule, including inside `Sequential`) between devices

**Training**
- Manual training loop (torch-style), or keras-style `model.compile()` / `model.fit()` / `model.evaluate()` — both fully supported, `fit` is just the manual loop wrapped in a method
- `fit` takes raw arrays or a `DataLoader`, returns a per-epoch `history`, supports `validation_data` and `patience` (early stopping via the `EarlyStopping` utility)

**Quantization**
- `model.quant()` — post-training INT8 quantization of `Linear` layers: per-channel int8 weights + float scale, dynamic activation quantization, real `int8 @ int8` integer matmul, ~4x smaller checkpoints

**Optimization (`tinytensor.optim`)**
- `SGD` (with momentum), `AdamW` (decoupled weight decay)
- `StepLR` learning rate scheduler
- `clip_grad_norm_`

**Data (`tinytensor.data`)**
- `Dataset`, `TensorDataset`, `DataLoader` with shuffling and batching

**Utilities (`tinytensor.utils`)**
- `progress_bar` / `train_bar`, `EarlyStopping`, `summary()` (architecture + parameter count printout)

**Model persistence**
- `Module.save()` / `Module.load()` — pickle-based `state_dict`, `.tt` extension, works recursively through nested modules and `Sequential`

## Project layout

```
tinytensor/
├── tinytensor/
│   ├── core/        # Tensor, autograd engine, ops
│   ├── nn/          # layers, activations, losses
│   ├── optim/       # SGD, AdamW, StepLR
│   ├── data/        # Dataset, DataLoader
│   ├── backends/    # cpu_numpy.py + cuda_gpu.cu / cuda_binding.cpp (cuBLAS)
│   ├── utils/       # progress_bar, EarlyStopping, summary
│   └── config.py    # random seed helper
├── examples/
├── tests/
├── docs/
├── setup.py
└── requirements.txt
```

## Known limitations

- `MultiHeadAttention` covers self-attention; there's no cross-attention variant, no relative/rotary position encodings, and `train_gpt.py` uses plain learned positional embeddings.
- `Conv2d`/`Sequential`/`RNN` forward passes are implemented directly against array data rather than composed purely out of `Tensor` operations, so each of them ships its own manually written `backward()` closure — verified against numerical gradient checking, but this means adding a new layer requires writing its gradient by hand rather than getting it for free from existing ops.
- `backward()` rebuilds the computation graph's topological order from scratch on every call (same approach as micrograd) — there is no graph caching or `retain_graph` equivalent.
- GPU support depends on `cupy` being installed and importable at runtime; the bundled `cuBLAS`/`pybind11` extension built at install time is only used as an availability signal (`HAS_CUDA`), actual GPU compute goes through `cupy`'s own kernels.
- Single-threaded `DataLoader`, no multiprocessing/prefetching.

## References

- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd)
- [Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [CS231n — Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- [Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [CuPy documentation](https://docs.cupy.dev/en/stable/)
