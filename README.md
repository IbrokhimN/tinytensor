# tinytensor

A small, dependency-light autograd engine and neural network toolkit built on
top of NumPy, with an optional CUDA backend via CuPy for real GPU-resident
tensors. It implements reverse-mode automatic differentiation (the same core
idea as [micrograd](https://github.com/karpathy/micrograd)), a standard set of
layers, optimizers, a data pipeline, ready-made model architectures, and basic
deployment tooling (quantization, pruning, ONNX export).

It is **not** a PyTorch replacement in performance or ecosystem. It exists to be
small enough to read end to end in an evening, while still being capable enough
to train real convolutional models on real data (MNIST, Fashion-MNIST) on both
CPU and GPU.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Training on the GPU](#training-on-the-gpu)
- [What's implemented](#whats-implemented)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
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
pip install -e .[dev]     # + pytest for the test suite
```

The only hard dependency is `numpy`. For GPU support, install `cupy` matching
your CUDA version, for example:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda13x   # CUDA 13.x
```

If `cupy` is not installed, everything falls back to a pure-NumPy CPU path — the
install never fails because of a missing CUDA toolchain. See
[docs/cuda.md](docs/cuda.md).

## Quickstart

Train a small CNN on real MNIST digits in a few lines:

```python
from tinytensor.data import load_mnist
from tinytensor.models import LeNet
from tinytensor.optim import AdamW
from tinytensor.nn import CrossEntropyLoss

# load_mnist downloads, caches, normalizes, and adds the channel dim -> [N,1,28,28]
(x_train, y_train), (x_test, y_test) = load_mnist()

model = LeNet(num_classes=10, in_channels=1)
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test),
)
```

`fit` prints a progress bar, per-epoch loss, training accuracy, and validation
loss, and returns a `history` dict.

## Training on the GPU

There is a single global switch. Call it once at the top of your script and
every new tensor, layer, and model is created on the GPU automatically — no
manual `.to("cuda")` on each object:

```python
import tinytensor as tt
tt.set_device("cuda")   # everything below is created on the GPU

# ... exact same training code as above ...
```

Device flows through the graph: each layer creates its output on the same device
as its input, so you set the device once and it propagates. `.to("cpu")` /
`.to("cuda")` still exist for moving individual tensors by hand when you want to.

`tt.cuda_available()` returns whether CuPy was found. Verified working on an
RTX 2080 (CUDA 13.3, `cupy-cuda13x`) training ResNet on MNIST end to end.

## What's implemented

### Core

- `Tensor` with reverse-mode autograd: `+ - * @ **`, `.sum()`, `.reshape()`,
  `.transpose()`, `.abs()`, `.log()`, broadcasting-aware gradients, correct
  gradients through batched (4D+) matmul.
- Per-tensor backend resolution via `get_array_module()` — every op dispatches
  to `numpy` or `cupy` based on where the data lives, so gradients stay on the
  correct device throughout a step.
- Activations with gradients: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`.

### Layers (`tinytensor.nn`)

- `Linear`, `Sequential`.
- `Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `Flatten`,
  `BatchNorm2d` — full backward via cached im2col/col2im, gradient-checked.
- `ResidualBlock` — the ResNet building block, with automatic 1x1 downsample
  when shapes change.
- `LayerNorm`, `MultiHeadAttention` (with causal masking), `Embedding`,
  `RNNCell`, `RNN` (full BPTT).
- Activations: `ReLU`, `LeReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`.
- `Dropout` — inverted dropout, gated by `train()`/`eval()`.
- Losses: `MSELoss`, `CrossEntropyLoss`, `BCELoss`.

### Model architectures (`tinytensor.models`)

- `LeNet` — classic CNN for 28x28 inputs (~62k params).
- `VGG` — VGG-style blocks for 32x32 inputs (~3.8M params).
- `ResNet`, `ResNet18`, `ResNet34` — residual networks with a `small_input`
  mode for CIFAR/MNIST-sized images (ResNet18 ~11.2M params).

### Data (`tinytensor.data`)

- `load_mnist`, `load_fashion` — download, cache, and optionally normalize
  (`normalize=True` returns `[N,1,28,28]` float32 in `[0,1]`).
- `Dataset`, `TensorDataset`, `DataLoader` (shuffling, batching, device-aware).
- Augmentations: `random_flip`, `random_crop`, `random_rotate90`, `add_noise`,
  `random_brightness`, and `Compose` to chain them.

### Training

- Manual loop (torch-style), or keras-style `compile` / `fit` / `evaluate`.
- `fit` accepts raw arrays or a `DataLoader`, returns per-epoch `history`
  (`loss`, `acc`, and `val_loss` when validation data is given), shows a
  progress bar and training accuracy, and supports early stopping via `patience`.

### Optimization (`tinytensor.optim`)

- `SGD` (momentum), `AdamW` (decoupled weight decay).
- `StepLR`, `CosineAnnealingLR` schedulers.
- `clip_grad_norm_`.

### Deployment

- `model.quant()` — post-training INT8 quantization of `Linear` layers
  (per-channel int8 weights, dynamic activation quant, real integer matmul,
  ~4x smaller checkpoints).
- `Sequential.prune(amount)` — structured pruning of the weakest neurons.
- `Sequential.to_onnx(path, input_dim)` — export Linear/ReLU stacks to ONNX,
  verified against onnxruntime.
- `Module.save()` / `load()` — pickle-based `state_dict`, `.tt` files.

### Performance

- im2col/col2im index caching — the index arrays depend only on shape and conv
  params, not on the data, so they are computed once per shape and reused. Roughly
  2x faster im2col; results are bit-for-bit identical (verified).

## Documentation

- [docs/getting_started.md](docs/getting_started.md) — install, quickstart, first loop
- [docs/tensor_and_autograd.md](docs/tensor_and_autograd.md) — how `Tensor` and `backward()` work
- [docs/nn.md](docs/nn.md) — every layer, activation, and loss, PyTorch-style reference
- [docs/optim.md](docs/optim.md) — optimizers and schedulers
- [docs/data.md](docs/data.md) — datasets, loaders, augmentations
- [docs/training.md](docs/training.md) — manual loop and keras-style API
- [docs/cuda.md](docs/cuda.md) — the CUDA backend and `set_device`
- [docs/quantization.md](docs/quantization.md) — INT8 quantization
- [docs/utils.md](docs/utils.md) — progress bars, early stopping, summary
- [docs/model_saving.md](docs/model_saving.md) — save/load format
- [docs/faq.md](docs/faq.md) — issues that came up during development

## Project layout

```
tinytensor/
├── tinytensor/
│   ├── core/        # Tensor, autograd engine, ops
│   ├── nn/          # layers, activations, losses, functional (im2col)
│   ├── models/      # LeNet, VGG, ResNet
│   ├── optim/       # SGD, AdamW, schedulers
│   ├── data/        # Dataset, DataLoader, dataset loaders, augmentations
│   ├── utils/       # progress bar, EarlyStopping, summary
│   └── config.py    # seed + global device
├── examples/
├── tests/
├── docs/
└── setup.py
```

## Known limitations

These are real and worth knowing before you rely on the library.

- **Speed.** The NumPy/CuPy backend goes through im2col for convolutions, which
  allocates large temporary matrices. Even on GPU this is far slower than
  cuDNN-based frameworks — ResNet18 on MNIST runs, but at roughly 2 minutes per
  epoch on an RTX 2080, not seconds. Fine for learning, not for serious training.
- **Hand-written backward per layer.** `Conv2d`, `BatchNorm2d`, pooling, `RNN`,
  and attention implement their forward against raw array data and ship their own
  manually written `backward()`. They are gradient-checked, but adding a new such
  layer means deriving its gradient by hand rather than getting it for free.
- **Tensor op coverage is limited.** No `sum(axis=...)`, no `mean`, no `clip`,
  no `exp`/`max` as autograd ops on `Tensor`. Some losses/layers reach into
  `.data` and NumPy directly as a result.
- **`BCELoss` has no `log(0)` guard yet.** The `eps` constructor argument exists
  but is not wired in; feeding exactly 0 or 1 probabilities can produce infinities.
- **Autograd graph is rebuilt every backward** via recursive topological sort
  (micrograd-style). No graph caching, no `retain_graph`, no `no_grad()` context —
  inference still builds the graph. Very deep graphs can hit Python's recursion limit.
- **ONNX export and pruning are narrow.** `to_onnx` covers Linear + ReLU stacks;
  other activations are silently skipped. `prune`/`to_onnx` assume a `Sequential`
  of `Linear` layers. Quantization is `Linear`-only and CPU-focused.
- **No CIFAR loader.** The official CIFAR source isn't reachable from the sandbox
  used during development; only MNIST and Fashion-MNIST loaders ship.
- **Single-threaded `DataLoader`** — no multiprocessing or prefetching.

## Roadmap

Things that are planned or would be natural next steps:

- CIFAR-10/100 loaders.
- A minimal GPT (the building blocks — `Embedding`, causal `MultiHeadAttention`,
  `LayerNorm` — are already here; positional encoding and a char/SentencePiece
  tokenizer are what's missing).
- Further GPU optimization beyond im2col caching (fewer temporary allocations).
- `no_grad()` context and an iterative (non-recursive) topological sort.
- More Tensor ops (`sum(axis)`, `mean`, `clip`) so fewer layers need hand-written backward.

## References

- [Andrej Karpathy — micrograd](https://github.com/karpathy/micrograd)
- [CS231n — Backpropagation](https://cs231n.github.io/optimization-2/)
- [Loshchilov & Hutter — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [He et al. — Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [CuPy documentation](https://docs.cupy.dev/en/stable/)
