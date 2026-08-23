# CUDA Backend

## How it works

There are two separate pieces, and it's worth keeping them distinct:

1. **Build-time**: `setup.py` checks for `nvcc`, then looks for `libcudart`/`libcublas` in a handful of standard locations (`$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda`, `$CONDA_PREFIX`). If both are found, it compiles `tinytensor.cuda_ops`, a small `pybind11` extension (`tinytensor/backends/cuda_gpu.cu` + `cuda_binding.cpp`). If anything is missing, the build silently falls back to a CPU-only build — the install never fails because of an incomplete CUDA toolchain.

2. **Runtime**: `tinytensor/core/tensor.py` tries `import cuda_ops` and `import cupy`. Both need to succeed for `HAS_CUDA = True`. `cupy` is what actually does the work — every GPU tensor's `.data` is a real `cupy.ndarray`, and every operation dispatches through `get_array_module(data)` (returns `cupy` or `numpy` based on the array's own type) rather than going through the compiled `cuda_ops` extension. In practice `cuda_ops` currently just acts as an availability signal alongside `cupy` in the `HAS_CUDA` check; the actual matrix multiplication and every other op run through `cupy`'s own kernels, which already use cuBLAS internally.

```python
x = Tensor([[1.0, 2.0]]).to("cuda")
w = Tensor([[1.0], [1.0]]).to("cuda")
out = x @ w   # x.data and w.data are cupy arrays; matmul runs via get_array_module -> cupy.matmul
```

`Module.to("cuda")` / `.cuda()` recursively convert every parameter's `.data` (and `.grad`, if present) to `cupy` arrays, walking into every submodule including layers stored inside `Sequential`.

## If the build fails

The most common failure is `nvcc` being present (e.g. installed via conda) while `libcudart`/`libcublas` are not physically installed anywhere the linker checks. This used to hard-fail the entire `pip install`; the current `setup.py` catches build failures for the CUDA extension specifically and falls back to a CPU-only install with a warning printed to the console — no reinstall needed, it just won't have the extension built (which, per above, matters less than it used to since `cupy` does the real compute work).

To actually get GPU acceleration working you need `cupy` installed and importable at runtime — that's the part that matters most:

```bash
pip install cupy-cuda12x   # match the suffix to your installed CUDA version
```

If you also want the build-time extension to compile (currently only used as part of the `HAS_CUDA` gate), install the missing CUDA libraries. Via conda:

```bash
conda install -c conda-forge cuda-cudart-dev libcublas-dev -y
```

Then rebuild:

```bash
pip uninstall pytinytensor -y
pip install pytinytensor --no-cache-dir
```

## Further reading

- [CuPy documentation](https://docs.cupy.dev/en/stable/) — the library actually doing the GPU compute
- [cuBLAS docs (cublasSgemm)](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
- [pybind11 docs](https://pybind11.readthedocs.io/en/stable/) — how the build-time extension is exposed to Python

## Publishing a wheel with CUDA support

A locally built `.whl` with the extension already compiled will **not** be accepted by PyPI as-is — bare Linux wheels (tagged `linux_x86_64`) are rejected; a `manylinux_*` tag via `auditwheel` is required, and `cudart`/`cublas` are not part of the standard manylinux image, so they'd need to be bundled in manually. The simplest path is publishing only the sdist (`.tar.gz`) — every user's `pip install` then compiles it locally against whatever CUDA toolchain (or lack of one) they actually have, and installs `cupy` separately as needed.
