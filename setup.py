import os
import subprocess
import pybind11
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

long_description = Path(__file__).parent / "README.md"
long_description = long_description.read_text(encoding="utf-8") if long_description.exists() else ""

def find_cuda_lib_dirs():
    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(conda_prefix)

    found = []
    for base in candidates:
        if not base:
            continue
        lib_dir = os.path.join(base, "lib64")
        if not os.path.isdir(lib_dir):
            lib_dir = os.path.join(base, "lib")
        has_cudart = any(
            f.startswith("libcudart.so") for f in os.listdir(lib_dir)
        ) if os.path.isdir(lib_dir) else False
        has_cublas = any(
            f.startswith("libcublas.so") for f in os.listdir(lib_dir)
        ) if os.path.isdir(lib_dir) else False
        if has_cudart and has_cublas:
            found.append(lib_dir)
    return found

def check_cuda():
    try:
        subprocess.check_output(["nvcc", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False, []

    lib_dirs = find_cuda_lib_dirs()
    if not lib_dirs:
        print("nvcc есть но libcudart/libcublas не найдены рядом. Сборка пойдет на CPU.")
        return False, []
    return True, lib_dirs


class CUDABuildExt(build_ext):
    # если cuda-сборка все равно упадет (например линковка) - не роняем весь pip install,
    # а просто откатываемся на чистый cpu-пакет
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile

        def custom_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # extra_postargs может приходить как список или словарь — обрабатываем оба случая
            if isinstance(extra_postargs, dict):
                gcc_postargs = extra_postargs.get('gcc', [])
                nvcc_postargs = extra_postargs.get('nvcc', [])
            else:
                gcc_postargs = extra_postargs or []
                nvcc_postargs = []

            if os.path.splitext(src)[1] == '.cu':
                # nvcc собирает .cu файл
                inc_flags = [f"-I{inc}" for inc in self.compiler.include_dirs]
                cmd = ['nvcc', '-c', src, '-o', obj] + inc_flags + nvcc_postargs
                self.spawn(cmd)
            else:
                # g++ собирает .cpp файл (с pybind11)
                original_compile(obj, src, ext, cc_args, gcc_postargs, pp_opts)

        self.compiler._compile = custom_compile

        try:
            super().build_extensions()
        except Exception as e:
            print(f"сборка cuda-расширения не удалась ({e}), продолжаем без него (cpu-only).")
            self.extensions = []

ext_modules = []

cuda_ok, cuda_lib_dirs = check_cuda()

if cuda_ok:
    cuda_module = Extension(
        'tinytensor.cuda_ops',
        sources=[
            'tinytensor/backends/cuda_gpu.cu',       # nvcc
            'tinytensor/backends/cuda_binding.cpp'   # g++
        ],
        include_dirs=[
            pybind11.get_include(),
            '/usr/local/cuda/include',
            os.path.expanduser('~/.local/include')
        ],
        library_dirs=cuda_lib_dirs,
        libraries=['cudart', 'cublas'],
        extra_compile_args={
            'gcc': ['-O3', '-fPIC', '-std=c++17'],
            'nvcc': ['-O3', '-Xcompiler', '-fPIC']
        },
        optional=True,
    )
    ext_modules.append(cuda_module)
    print("cublas")
else:
    print("cpu")

setup(
    name="pytinytensor",
    version="0.1.6",
    description="мини ИИ фреймворк от IbrokimN ( github/IbrokhimN )",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="IbrokhimN",
    url="https://github.com/IbrokhimN/tinytensor",
    packages=find_packages(include=["tinytensor", "tinytensor.*"]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDABuildExt},
    install_requires=[
        "numpy>=1.23",
        "pybind11>=2.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    python_requires=">=3.9",
)
