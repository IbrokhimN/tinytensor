# Model Saving

## Usage

```python
model.save("model.tt")
model.load("model.tt")
```

The `.tt` extension is just a convention — the file is a plain pickle of `state_dict()`, a `{parameter_name: array}` dict.

## How it works internally

```python
def state_dict(self):
    sdict = {}
    for name, param in self._get_named_params().items():
        sdict[name] = param.data
    return sdict

def load_state_dict(self, sdict):
    for name, param in self._get_named_params().items():
        if name in sdict:
            param.data = sdict[name]

def save(self, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(self.state_dict(), f)

def load(self, filepath):
    with open(filepath, "rb") as f:
        sd = pickle.load(f)
        self.load_state_dict(sd)
```

`_get_named_params` recursively walks nested `Module`s, producing dotted names like `fc1.weight`, `fc1.bias`, `fc2.weight`, so a multi-layer model saves and loads in one call. `Sequential` overrides `_get_named_params` explicitly (it stores layers in a `self.layers` list, which the base implementation does not know how to walk into — without the override, `state_dict()` on a `Sequential` silently returns an empty dict and `load()` becomes a silent no-op).

## Saving a GPU-resident model

`param.data` for a `cuda` tensor is a `cupy` array; pickling `cupy` arrays directly works but ties the checkpoint to having `cupy`/CUDA available wherever it's loaded back. Move the model to CPU before saving if you want a portable checkpoint:

```python
model.cpu()
model.save("model.tt")
```

## Gotchas already hit during development

File mode — easy to get backwards:

- `save` writes to the file → needs `"wb"` (write binary)
- `load` reads from the file → needs `"rb"` (read binary)

Getting it backwards either raises `io.UnsupportedOperation` immediately, or — worse — `load()` opened in `"wb"` silently truncates the file before `pickle.load` even runs, so it fails on an empty file with a confusing error.

Missing `return sdict` at the end of `state_dict()` is the other one that bit us: the function builds the dict correctly but forgets to return it, so `state_dict()` returns `None` and `save()` writes a pickled `None` to disk with zero errors raised anywhere. Sanity check after any change to this code path:

```python
before = model.weight.data.copy()
model.save("test.tt")
model2 = SameArchitecture()
model2.load("test.tt")
assert (before == model2.weight.data).all()
```

## Limitations

- Optimizer state (`AdamW`'s `m`/`v` buffers, `SGD`'s momentum buffers) is **not** saved, only model weights. Resuming training after a load restarts the optimizer from a clean state.
- The format is not versioned. Loading a `.tt` file into a model with a different architecture silently ignores any parameter names that don't match (`if name in sdict`) — no warning, no error. Track architecture versions yourself if that matters for your use case.
