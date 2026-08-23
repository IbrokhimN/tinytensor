# сравнение скорости на GPU: tinytensor (CuPy) vs pytorch (cuDNN).
# на GPU операции асинхронные, поэтому ОБЯЗАТЕЛЬНА синхронизация перед замером,
# иначе намеряешь только время запуска, а не реального счёта.
# нужен torch с cuda + cupy: pip install torch, pip install cupy-cuda12x
import time
import numpy as np

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
    if not HAS_TORCH:
        print("torch есть, но CUDA недоступна - проверь установку GPU-версии")
except ImportError:
    HAS_TORCH = False
    print("torch не установлен")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("cupy не установлен - tinytensor не сможет на GPU. поставь cupy-cuda12x")

import tinytensor as tt
from tinytensor.core.tensor import Tensor
from tinytensor.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from tinytensor.optim import AdamW

if not HAS_CUPY:
    print("без cupy GPU-бенчмарк невозможен. выход.")
    raise SystemExit

tt.set_device("cuda")


def sync_tt():
    # ждём пока GPU реально досчитает (cupy)
    cp.cuda.runtime.deviceSynchronize()


def sync_pt():
    torch.cuda.synchronize()


def timeit_tt(fn, runs=30, warmup=10):
    for _ in range(warmup):
        fn()
    sync_tt()
    ts = []
    for _ in range(runs):
        sync_tt()
        t0 = time.perf_counter()
        fn()
        sync_tt()               # ждём завершения ПЕРЕД остановкой таймера
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def timeit_pt(fn, runs=30, warmup=10):
    for _ in range(warmup):
        fn()
    sync_pt()
    ts = []
    for _ in range(runs):
        sync_pt()
        t0 = time.perf_counter()
        fn()
        sync_pt()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def bench_matmul():
    a = np.random.randn(1024, 1024).astype(np.float32)
    b = np.random.randn(1024, 1024).astype(np.float32)
    ta, tb = Tensor(a), Tensor(b)
    tt_time = timeit_tt(lambda: ta @ tb)
    if HAS_TORCH:
        pa = torch.from_numpy(a).cuda()
        pb = torch.from_numpy(b).cuda()
        pt_time = timeit_pt(lambda: pa @ pb)
    else:
        pt_time = None
    return tt_time, pt_time


def bench_conv():
    x = np.random.randn(32, 3, 32, 32).astype(np.float32)
    conv_tt = Conv2d(3, 16, 3, padding=1)
    tx = Tensor(x)
    tt_time = timeit_tt(lambda: conv_tt(tx))
    if HAS_TORCH:
        import torch.nn as tnn
        conv_pt = tnn.Conv2d(3, 16, 3, padding=1).cuda()
        px = torch.from_numpy(x).cuda()
        with torch.no_grad():
            pt_time = timeit_pt(lambda: conv_pt(px))
    else:
        pt_time = None
    return tt_time, pt_time


def bench_train():
    x = np.random.randn(64, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 64)
    m = Sequential(
        Conv2d(1, 6, 5, padding=2), ReLU(), MaxPool2d(2),
        Conv2d(6, 16, 5), ReLU(), MaxPool2d(2),
        Flatten(), Linear(16 * 5 * 5, 10),
    )
    opt = AdamW(m.parameters(), lr=1e-3)
    lossf = CrossEntropyLoss()
    tx, ty = Tensor(x), Tensor(y)

    def tt_step():
        pred = m(tx)
        loss = lossf(pred, ty)
        opt.zero_grad()
        loss.backward()
        opt.step()
    tt_time = timeit_tt(tt_step, runs=15, warmup=5)

    if HAS_TORCH:
        import torch.nn as tnn
        pm = tnn.Sequential(
            tnn.Conv2d(1, 6, 5, padding=2), tnn.ReLU(), tnn.MaxPool2d(2),
            tnn.Conv2d(6, 16, 5), tnn.ReLU(), tnn.MaxPool2d(2),
            tnn.Flatten(), tnn.Linear(16 * 5 * 5, 10),
        ).cuda()
        popt = torch.optim.AdamW(pm.parameters(), lr=1e-3)
        plossf = tnn.CrossEntropyLoss()
        px = torch.from_numpy(x).cuda()
        py = torch.from_numpy(y).long().cuda()

        def pt_step():
            pred = pm(px)
            loss = plossf(pred, py)
            popt.zero_grad()
            loss.backward()
            popt.step()
        pt_time = timeit_pt(pt_step, runs=15, warmup=5)
    else:
        pt_time = None
    return tt_time, pt_time


def show(name, a, b):
    if b is not None:
        print(f"{name:20} tinytensor={a:8.2f}мс  pytorch={b:8.2f}мс  медленнее в {a/b:.1f}x")
    else:
        print(f"{name:20} tinytensor={a:8.2f}мс  (torch/cuda недоступен)")


print("=" * 70)
print("Сравнение скорости на GPU: tinytensor (CuPy) vs pytorch (cuDNN)")
print("=" * 70)
show("matmul 1024x1024", *bench_matmul())
show("conv2d forward", *bench_conv())
show("train step (LeNet)", *bench_train())
print("=" * 70)
print("на GPU разрыв должен быть меньше, чем на CPU: обе библиотеки используют")
print("оптимизированные ядра NVIDIA (cuBLAS/cuDNN). matmul почти вровень.")
