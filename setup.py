import os
import subprocess
import pybind11
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

def check_cuda():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

class CUDABuildExt(build_ext):
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
        super().build_extensions()

ext_modules = []

if check_cuda():
    cuda_module = Extension(
        'tinytensor.cuda_ops',
        sources=[
            'tinytensor/backends/cuda_gpu.cu',       # Компилируется nvcc
            'tinytensor/backends/cuda_binding.cpp'    # Компилируется g++
        ],
        include_dirs=[
            pybind11.get_include(),
            '/usr/local/cuda/include',
            os.path.expanduser('~/.local/include')
        ],
        library_dirs=['/usr/local/cuda/lib64'],
        libraries=['cudart', 'cublas'],
        extra_compile_args={
            'gcc': ['-O3', '-fPIC', '-std=c++17'],
            'nvcc': ['-O3', '-Xcompiler', '-fPIC']
        }
    )
    ext_modules.append(cuda_module)
    print("🔥 CUDA найдена! Собираем cuBLAS бэкенд...")
else:
    print("⚠️ CUDA не найдена. Сборка продолжится на CPU (NumPy).")

setup(
    name="tinytensor",
    version="0.1.0",
    description="мини ИИ фреймворк от IbrokimN ( github/IbrokhimN )",
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