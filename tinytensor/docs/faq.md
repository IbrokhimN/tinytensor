# FAQ / Known Gotchas

A running list of everything that has already gone wrong during development, kept here so it doesn't have to be rediscovered twice.

## `ModuleNotFoundError: No module named 'pybind11'` when running `python3 -m build`

`setup.py` does `import pybind11` at the top of the file, and `pip`/`build` create an **isolated build environment** that by default only contains `setuptools`. A `pyproject.toml` next to `setup.py` is required:

```toml
[build-system]
requires = ["setuptools>=61", "wheel", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"
```

Without this file (or if it goes missing after moving to a new machine/OS), the build fails with this exact error every time.

## `pip install pytinytensor` fails with `cannot find -lcudart` / `-lcublas`

`nvcc` was found but `libcudart`/`libcublas` are not physically installed anywhere the linker checks (common with conda setups where `nvcc` is installed separately from the runtime libraries). See [cuda.md](cuda.md) — the current `setup.py` no longer hard-fails on this, it falls back to a CPU-only install with a warning.

## `save`/`load` don't seem to do anything, or crash

Classic mixed-up file mode. `save` = write = `"wb"`, `load` = read = `"rb"`. Details and a sanity-check snippet in [model_saving.md](model_saving.md).

## `Dropout` / a custom layer raises `AttributeError: 'X' object has no attribute 'training'`

Forgot to call `super().__init__()` in the layer's constructor. `self.training` only exists after `Module.__init__` runs.

## The model behaves the same on inference as during training (dropout keeps zeroing activations)

Did you call `model.eval()` before inference? `train()`/`eval()` recursively set `self.training` on every submodule, but if layers are called by hand outside the normal `forward()` chain, double-check the flag actually landed on the right object.

## A layer's output has `_prev == set()` and gradients never reach it

Happens when a new layer's forward pass is written directly against `.data` (rather than composed out of existing `Tensor` operations, which wire the graph automatically) and forgets to set `out._prev`/`out._backward` on its output. Calling `.backward()` on anything downstream then simply won't reach that layer's parameters or its input — no error, no warning, the weights just never update. Has hit `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d`, `Softmax`, `CrossEntropyLoss`, `Embedding`, and `RNNCell`/`RNN` at various points before being fixed. When writing a new layer that touches `.data` directly: always set `_prev` and a `_backward` closure on the output, and verify with numerical gradient checking (perturb an input by `±eps`, compare to the analytic gradient) — a crash-free forward pass proves nothing about whether backward actually works.

## `RNN`'s output looks connected but `backward()` crashes with `UFuncTypeError` (adding `None` to an array)

A version of `RNN.forward` linked the stacked output's `_prev` only to the raw input `x`, instead of to the set of intermediate per-timestep hidden states. The `_backward` closure then tried to read `.grad` off those intermediate tensors — but since they were never actually part of the graph (`out._prev` didn't include them), nothing had ever set their `.grad`, so it was still `None` when read. `out._prev` for a sequence-stacking op needs to be the full set of tensors it was assembled from, not just the original input.

## A closure inside a loop always uses the *last* loop value, not the one from its own iteration

Python's late-binding closures: a nested function that references a loop variable directly (rather than receiving it as an argument) sees whatever that variable is at **call time**, not at the time the closure was created. This broke `RNN`'s backpropagation-through-time once: a `_backward` closure created inside a `for t in range(seq_len)` loop referenced the loop variable directly, so every timestep's closure ended up seeing the same, final value — collapsing all timesteps' input gradients into one. Fixed by passing the loop variable explicitly into a factory function:

```python
def make_backward(t, x_t):
    def _backward():
        x.grad[:, t, :] += x_t.grad
    return _backward

x_t._backward = make_backward(t, x_t)  # bound by value, not by reference
```

Any future code that builds closures inside a loop over timesteps/layers/batches should watch for this specifically.

## `Sigmoid` import errors breaking `tinytensor.nn` entirely

The activation class used to be misspelled `Sigmod`, and `nn/__init__.py` regressed to importing the correctly-spelled `Sigmoid` more than once while the class itself still said `Sigmod`. Because Python always executes a package's `__init__.py` before any of its submodules, a single bad import there breaks **every** import from `tinytensor.nn` at once — `Linear`, `Dropout`, both example scripts, most of the test suite, all of it. The class has since been renamed to `Sigmoid` for good (fixing the typo at the source rather than just patching the import), but if `tinytensor.nn` suddenly can't be imported at all, check `nn/__init__.py`'s import line against the actual class name in `activations.py` first — it's happened enough times to be the first thing worth checking.

## `AttributeError: 'Tensor' object has no attribute 'device'` the moment CUDA/`cupy` become available

`Tensor.__init__` referenced `self.device` in a conditional (`elif HAS_CUDA and self.device == 'cuda':`) before `self.device` had actually been assigned. With `HAS_CUDA=False` (no `cupy` installed) Python short-circuits the `and` and never evaluates the right side, so this stayed hidden in any environment without a working `cupy`+CUDA setup — and then broke *every single* `Tensor(...)` call the moment both were actually available, which is exactly the environment this code path exists for. Fixed by setting `self.device = device` as the very first line of `__init__`.

## `Tensor(some_other_tensor)` raises `TypeError: float() argument must be a string or a real number, not 'Tensor'`

`__init__` used to run a generic `np.array(data, dtype=np.float32)` conversion *before* checking `isinstance(data, Tensor)`, so wrapping an existing `Tensor` tried to convert the `Tensor` object itself into a NumPy array and failed. The `isinstance(data, Tensor)` branch needs to run first, with nothing before it that touches `data` unconditionally.

## `Tensor` stopped enforcing `float32` for input NumPy arrays

A duplicated code path in `__init__` cast to `float32` in one branch and then immediately got overwritten by a second, later branch that assigned the raw array back without casting — so `Tensor(np.array([1,2,3], dtype=np.int32)).data.dtype` came back `int32` instead of `float32`. Caught by `test_creation_from_list_and_ndarray`. The fix was removing the redundant first branch entirely rather than patching both.

## `get_array_module` never detects `cupy`, always falls back to `numpy`

```python
def get_array_module(data):
    if hasattr(data, "__module__") and "cupy" in data.__module__:
        ...
```

`__module__` lives on a *type*, not on an array instance — `hasattr(some_array, "__module__")` is `False` for both `numpy` and `cupy` arrays alike, so this check silently never took the `cupy` branch, for anyone. Fixed by checking `type(data).__module__` instead of `data.__module__`.

## `Sequential.to("cuda")` doesn't raise anything, but nothing moves to the GPU

Same root cause as the `parameters()`/`_get_named_params()` issue above: `Sequential` stores its layers in `self.layers`, a plain list, and the base `Module.to()` only scans direct `__dict__` attributes for `Tensor`/`Module` instances — it doesn't know how to look inside a list. Without an explicit `Sequential.to()` override that walks `self.layers`, calling `.to("cuda")` (or `.cuda()`) on a `Sequential` model returns successfully having moved exactly nothing.

## Unexpected duplicate class definitions, like two different `Sequential`s

At one point there was both `tinytensor/nn/modules.py::Sequential` (the one actually imported by `nn/__init__.py`, with the `_get_named_params`/`to()` overrides described above) and a separate, unused `tinytensor/nn/sequential.py::Sequential` missing those same overrides. The second file wasn't wired into the public API and so didn't cause a live bug, but it's exactly the kind of landmine that goes off the moment someone switches the import to point at it without noticing the missing fixes. Worth grepping for duplicate class names occasionally:

```bash
grep -rn "^class Sequential" tinytensor/
```

## Unexpected nested copy of the package in `git`, like `tinytensor/tinytensor/tinytensor/...`

Happened once when `setup.py build` / `pip install -e .` was run from inside the package directory instead of the repo root, and the result got swept into `git add .`. Check:

```bash
git ls-files | grep "tinytensor/tinytensor/"
```

If anything shows up, it needs cleaning out (`git rm -r --cached`, update `.gitignore`, commit).

## `twine upload` fails with `Binary wheel ... has an unsupported platform tag 'linux_x86_64'`

PyPI rejects bare Linux wheels, only `manylinux_*` tags are accepted. Upload just the sdist:

```bash
twine upload dist/pytinytensor-X.Y.Z.tar.gz
```

More detail on packaging and CUDA wheels in [cuda.md](cuda.md#publishing-a-wheel-with-cuda-support).

## KDE/GNOME pops up a "wallet password" dialog during `twine upload`

`keyring` trying to store the token in the system password manager, unrelated to the upload itself. Cancel it; `twine` will fall back to a plain-text prompt. To make it stop asking: `pip uninstall keyring`.

## PyPI shows "The author of this package has not provided a project description"

Either `long_description` wasn't set in `setup.py` (needs to read `README.md` and set `long_description_content_type="text/markdown"`), or an old version's page is being viewed — check `pypi.org/project/pytinytensor/X.Y.Z/` with the actual current version number.

## Can't re-upload the same version to PyPI

By design, even after deleting the release. Bump the version (`0.1.1` → `0.1.2`) before every new upload.
