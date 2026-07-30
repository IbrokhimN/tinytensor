# quantization

Post-training INT8 quantization for `Linear` layers. Weights are stored as
`int8` plus a per-channel `float32` scale, and the matmul runs in integers.
This is real quantization — not fake/simulated — so the weights actually shrink
on disk and in memory (~4x), and the forward pass does integer `int8 @ int8`
arithmetic.

## Quickstart

```python
model = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
# ... train the model ...

model.quant()               # convert every Linear to int8 in place
pred = model(x)             # forward now runs the integer path
model.save("model_int8.tt") # checkpoint stores int8 weights + scales
```

`quant()` is recursive: it walks the whole module tree (including inside
`Sequential`) and quantizes every `Linear`. Layers without a `quant()` method of
their own (activations, `Conv2d`, `Embedding`, norms) are left untouched.

## The math

Quantization maps `float32` numbers onto the integer range `[-127, 127]`
(symmetric int8). For an array `W`:

```
scale = max(|W|) / 127
W_q   = clip(round(W / scale), -127, 127)   # int8
```

Dequantization is just the inverse: `W ≈ W_q * scale`. The `scale` is a single
float you store alongside the integers. The `round()` is where precision is
lost — usually around a 1% change in the output.

**Weights** are quantized *per-channel* (one scale per output neuron). Different
neurons have different weight magnitudes, so a shared scale would over-compress
the small ones; a per-neuron scale is much more accurate at almost no cost (one
extra float per neuron).

**Activations** are quantized *dynamically* at each forward — their range
depends on the input and isn't known ahead of time, so the scale is computed on
the fly (per-tensor).

The integer matmul works because the scales factor out of the sum:

```
x @ W ≈ (x_q · scale_x) @ (W_q · scale_w)
      = (x_q @ W_q) · scale_x · scale_w
```

So the forward pass is: quantize `x`, compute `x_q @ W_q` in `int32`, then
multiply the result by `scale_x * scale_w` and add the (float) bias.

## What gets quantized

Only `Linear`. In an MLP or transformer that's the bulk of the parameters, so
this is the standard choice. `bias` stays `float32` (it's tiny and quantizing it
would only add error). Because bias and scales stay float, the compression ratio
is a little under 4x on small layers and approaches 4x as layers grow.

```python
model.quant()
# fc1.weight  -> fc1.w_q (int8) + fc1.w_scale (float32, one per neuron)
# fc1.bias    -> fc1.b_data (float32, unchanged)
# the original float32 weight/bias are dropped to actually free memory
```

## Saving and loading

`save()`/`load()` understand quantized checkpoints. The state dict stores the
int8 buffers (`w_q`, `w_scale`, `b_data`) instead of float weights, so the file
is ~4x smaller. On `load()`, a layer that finds int8 buffers for itself switches
into quantized mode automatically — no extra call needed.

```python
model.quant()
model.save("model_int8.tt")

fresh = Sequential(Linear(784, 256), ReLU(), Linear(256, 10))
fresh.load("model_int8.tt")   # comes back already quantized
pred = fresh(x)
```

## Limitations

- CPU/NumPy only. Quantize on CPU: call `model.cpu()` before `model.quant()`.
  int8 on the GPU wouldn't be faster here without custom kernels.
- Post-training only — there is no quantization-aware training, and no
  `dequant()` to go back to float. Keep a float checkpoint from before
  `quant()` if you might want to fine-tune later.
- `int32` accumulation can only overflow at absurd `in_features` (>~130k), which
  you won't hit in practice.

> See [docs/nn.md](nn.md) for `Linear` internals and
> [docs/model_saving.md](model_saving.md) for the checkpoint format.
