# Cuda-бэкенд

## Как это работает

При установке `setup.py` сам проверяет, есть ли `nvcc`, и если да - ищет `libcudart`/`libcublas` в нескольких стандартных местах (`$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda`, `$CONDA_PREFIX`). Если нашел и то и другое - собирает `tinytensor.cuda_ops`, маленький pybind11-модуль, который дергает `cublasSgemm` (`tinytensor/backends/cuda_gpu.cu` + `cuda_binding.cpp`). Если чего-то не хватает - тихо откатывается на cpu-путь, установка не падает.

Сейчас куда греет только `matmul` - если у любого из двух тензоров `device == "cuda"` и модуль собрался, `Tensor.__matmul__` уходит в `cuda_ops.matmul` вместо numpy. Все остальные операции (сложение, активации и тд) все равно считаются на cpu через numpy, реального управления памятью на gpu тут нет - данные каждый раз гоняются host <-> device внутри одного вызова.

```python
x = Tensor([[1.0, 2.0]]).to("cuda")
w = Tensor([[1.0], [1.0]]).to("cuda")
out = x @ w   # уйдет в cublas, если бэкенд собрался
```

## Если сборка падает

Самая частая проблема - `nvcc` есть, а `libcudart`/`libcublas` физически не установлены (например nvcc стоит через conda, а сами cuda-либы нет). Раньше это роняло весь `pip install`, сейчас `setup.py` ловит такие ошибки сборки и просто откатывается на cpu-версию с предупреждением в консоли - переустанавливать ничего не надо, само разрешится, просто без ускорения.

Если хочется реально собрать gpu-версию и она не собирается - поставьте недостающие cuda-библиотеки. Через conda:

```bash
conda install -c conda-forge cuda-cudart-dev libcublas-dev -y
```

После этого пересоберите:

```bash
pip uninstall pytinytensor -y
pip install pytinytensor --no-cache-dir
```

## Что почитать

- [cuBLAS docs (cublasSgemm)](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
- [pybind11 docs](https://pybind11.readthedocs.io/en/stable/) - как из C++/CUDA дергать функции из питона

## Публикация wheel с кудой

Если собираете локально `.whl` с уже вкомпиленным `.so` - учтите, что PyPI **не примет** голый линуксовый wheel (`linux_x86_64`), нужен `manylinux_*` тег через `auditwheel`, а `cudart`/`cublas` не входят в стандартный manylinux-образ, их придется тащить самому. Проще всего публиковать только sdist (`.tar.gz`) - тогда пакет соберется у каждого пользователя локально под его окружение (плюс/минус то, что уже описано выше).
