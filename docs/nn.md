# nn

## Module

Every layer inherits from `Module`. Core pieces:

```python
from tinytensor.nn import Module

class MyLayer(Module):
    def __init__(self):
        super().__init__()   # required, or self.training never gets set
        ...

    def forward(self, x):
        ...
```

- `forward(*args, **kwargs)` — the layer's logic; override this in every subclass
- `__call__` — just calls `forward`, so `model(x)` and `model.forward(x)` are equivalent
- `parameters()` — recursively collects every `Tensor` with `requires_grad=True`, walking into nested `Module`s
- `zero_grad()` — resets `.grad` to `None` on every parameter
- `train(mode=True)` / `eval()` — recursively sets `self.training` on this module and every submodule (needed for `Dropout` and for `BatchNorm2d`'s running-stats behavior)
- `to(device)` / `.cuda()` / `.cpu()` — recursively moves every parameter and submodule between `numpy` and `cupy` storage
- `state_dict()` / `load_state_dict()` / `save()` / `load()` — see [model_saving.md](model_saving.md)

Modeled on [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), without hooks or buffers.

> If a custom layer holds nested `Module`s (e.g. `self.fc1 = Linear(...)`), always call `super().__init__()` in its constructor. Skipping it means `self.training` never exists, and anything relying on it (`Dropout`, `train()`/`eval()`) breaks with `AttributeError`.
>
> If a custom layer holds a *list* of nested `Module`s (like `Sequential` does with `self.layers`), the base `Module.parameters()`/`_get_named_params()`/`to()` only scan direct `__dict__` attributes — they don't look inside lists. `Sequential` explicitly overrides all three for this reason; any similar container-holding layer needs the same overrides, or its parameters silently won't be found, saved, or moved to the GPU.

## Sequential

```python
from tinytensor.nn import Sequential, Linear, ReLU

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)
```

`model.layers` holds the list; `model[i]` and `len(model)` both work. `parameters()`, `_get_named_params()` (used by `save`/`load`), and `to()` are all overridden explicitly to walk `self.layers` — see the note above for why.

## Linear

```python
from tinytensor.nn import Linear

layer = Linear(in_features=784, out_features=128)
out = layer(x)  # x.shape == (batch, 784) -> out.shape == (batch, 128)
```

`y = xW + b`. Weights are initialized with [He/Kaiming init](https://arxiv.org/abs/1502.01852) (`std = sqrt(2/in_features)`), a reasonable default for ReLU-family activations.

## Conv2d / MaxPool2d / AvgPool2d / BatchNorm2d / Flatten

```python
from tinytensor.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, Sequential

model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1),
    BatchNorm2d(8),
    ReLU(),
    MaxPool2d(2, 2),
    Flatten(),
    Linear(8 * 14 * 14, 10),
)
```

- `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` — implemented via im2col/col2im, not a naive sliding-window loop. Weight gradient, bias gradient, and input gradient are all hand-derived and verified against numerical gradient checking (max deviation ~1e-3 at float32 precision).
- `MaxPool2d(kernel_size, stride, padding=0)` — backward routes gradient only to the argmax location inside each pooling window.
- `AvgPool2d(kernel_size, stride, padding=0)` — backward distributes gradient uniformly across every position in the window.
- `BatchNorm2d(num_features, eps=1e-5, momentum=0.1)` — normalizes over `(batch, height, width)` per channel. Maintains `running_mean`/`running_var`, updated only in training mode (`self.training == True`); in eval mode the running statistics are used directly and the backward pass simplifies accordingly.
- `Flatten()` — reshapes `(N, C, H, W)` to `(N, C*H*W)`. Implemented via `Tensor.reshape()`, so it's fully differentiable through the reshape.

This stack (`Conv2d → BatchNorm2d → ReLU → MaxPool2d`, repeated, then `Flatten → Linear`) has been trained on real MNIST digits (~85% test accuracy in 5 epochs on a 4000-sample subset, on CPU).

## LayerNorm

```python
from tinytensor.nn import LayerNorm

ln = LayerNorm(normalized_shape=64, eps=1e-5)
out = ln(x)   # normalizes over the last dimension(s), not over the batch
```

Unlike `BatchNorm2d` (normalizes per-channel over the batch+spatial axes, needs running statistics), `LayerNorm` normalizes each sample independently over its own feature axis — same computation regardless of batch size, no running mean/var to track. This is the normalization transformers use, since attention/text sequences don't have the fixed spatial structure batchnorm assumes. `normalized_shape` can be an int (normalize the last axis) or a tuple (normalize the last N axes).

## MultiHeadAttention

```python
from tinytensor.nn import MultiHeadAttention

attn = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.1, causal=True)
out = attn(x)   # x: (batch, seq_len, embed_dim) -> out: (batch, seq_len, embed_dim)
```

Standard scaled dot-product attention: `Q`, `K`, `V` come from three separate `Linear` projections, `embed_dim` is split into `num_heads` heads of `embed_dim // num_heads` each (must divide evenly), scores are `Q @ Kᵀ / sqrt(head_dim)`, softmax, `@ V`, heads are merged back and passed through a final output `Linear`.

`causal=True` adds a large negative bias (`-1e9`) to the upper triangle of the attention scores before the softmax, so each position can only attend to itself and earlier positions — the standard GPT-style autoregressive mask. Pass an explicit `mask` tensor to `forward(x, mask=...)` for other masking patterns (e.g. padding masks); it's added to the raw scores the same way, before the softmax.

Batched multi-head attention needs correct gradients through 4D matrix multiplication (`(batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)`) and through splitting/merging heads (`Tensor.reshape()` + `Tensor.transpose()`). Both were gradient-checked directly through a full causal attention block, not just the individual ops in isolation.

```python
from tinytensor.nn import Embedding

emb = Embedding(vocab_size=1000, embed_dim=64)
out = emb(token_ids)  # token_ids: Tensor of integer indices, any shape -> out shape (*token_ids.shape, embed_dim)
```

A lookup table (`weight` of shape `(vocab_size, embed_dim)`) with a scatter-add backward: gradient for each row of `weight` is accumulated (`np.add.at`) from every position where that row's index was used, correctly handling repeated indices within a batch.

## RNNCell / RNN

```python
from tinytensor.nn import RNNCell, RNN

cell = RNNCell(input_size=8, hidden_size=16)
h_next = cell(x_t, h_prev)   # h_prev can be None on the first step, defaults to zeros

rnn = RNN(input_size=8, hidden_size=16)
out, h_final = rnn(x)   # x: (batch, seq_len, input_size) -> out: (batch, seq_len, hidden_size)
```

`RNNCell` implements `h_next = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)`, with a full hand-derived backward covering both weight matrices, both biases, the input, and the previous hidden state.

`RNN` loops `RNNCell` over the time dimension and performs full backpropagation through time (BPTT): every per-timestep input slice is kept in the graph (linked back to the original input tensor, not detached), and the final stacked output's `_prev` is the full set of intermediate per-timestep hidden states, so gradient reaches every timestep, not just the last one. `RNN.forward` returns a `(output, final_hidden_state)` tuple.

> `Tensor` does not yet have general `__getitem__` slicing. To use only the last timestep's output (typical for sequence classification/next-token prediction), slice `out.data` manually and re-wire `_prev`/`_backward` by hand — see the worked example in [getting_started.md](getting_started.md#a-recurrent-model). This is a known rough edge, not an intended permanent API.

There is no LSTM/GRU cell yet, only the plain tanh RNN cell.

## Activations

Thin wrappers over `Tensor` methods, each fully differentiable:

```python
from tinytensor.nn import ReLU, LeReLU, Sigmoid, Tanh, GELU, Softmax

ReLU()(x)
LeReLU(alpha=0.01)(x)
Sigmoid()(x)
Tanh()(x)
GELU()(x)
Softmax(dim=1)(x)
```

`Softmax` computes the full Jacobian-vector product on backward (`dx = y * (dy - sum(dy*y, axis=dim))`), not a shortcut — it is safe to use inside a network, not only as a final inference-time output.

## Dropout

```python
from tinytensor.nn import Dropout

drop = Dropout(p=0.5)
```

Inverted dropout: during training, a fraction `p` of activations is zeroed and the rest scaled by `1/(1-p)`. During eval it's a no-op. The random mask is generated with `get_array_module(x.data)`, so on a GPU tensor the mask is produced by `cupy`'s RNG directly rather than on the CPU and copied over.

> Without calling `model.eval()`, dropout stays active — there's no separate global switch; `self.training` lives on each module and `train()`/`eval()` recursively propagate it from the top-level model.

> Do not use `p=1.0` — the inverted-dropout scaling divides by `1-p`, so `p=1.0` is a division by zero and silently produces `NaN`.

## Loss functions

```python
from tinytensor.nn import MSELoss, CrossEntropyLoss

MSELoss()(pred, target)                 # mean((pred - target)^2)
CrossEntropyLoss()(logits, targets)     # targets: class indices (int) or one-hot
```

- `MSELoss` — for regression, or as a crude substitute for classification (works, converges worse than proper cross-entropy on many-class problems).
- `CrossEntropyLoss` — combined log-softmax + negative log-likelihood, implemented with the standard `(softmax(logits) - one_hot(target)) / batch_size` gradient, numerically stabilized with a max-subtraction before the exponential. `targets` can be given either as integer class indices (shape `(batch,)`) or as a one-hot / soft-label tensor of the same shape as `logits`.
