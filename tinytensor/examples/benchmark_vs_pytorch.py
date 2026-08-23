# сравнение скорости tinytensor и pytorch на CPU.
# честно: размеры достаточно большие чтобы доминировал реальный счёт,
# а не оверхед запуска. прогрев + median + несколько замеров.
# нужен torch: pip install torch
import time
import numpy as np

try:
    import torch
    import torch.nn as tnn
    torch.set_num_threads(torch.get_num_threads())  # используем все ядра, как numpy
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("torch не установлен. поставь: pip install torch")
    print("бенчмарк tinytensor всё равно прогоним\n")

from tinytensor.core.tensor import Tensor
from tinytensor.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from tinytensor.optim import AdamW
import tinytensor as tt

tt.set_device("cpu")


def timeit(fn, runs=30, warmup=15):
    # median время в мс. большой warmup чтобы убрать холодный старт
    # (первые вызовы медленнее: инициализация потоков, кэши)
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    # median устойчивее к выбросам чем среднее
    return float(np.median(ts))


def bench_matmul():
    # крупная матрица: тут доминирует реальный счёт, а не оверхед.
    # оба через BLAS (numpy) / MKL (torch), torch обычно чуть впереди
    a = np.random.randn(2048, 2048).astype(np.float32)
    b = np.random.randn(2048, 2048).astype(np.float32)
    ta, tb = Tensor(a), Tensor(b)
    tt_time = timeit(lambda: ta @ tb, runs=20, warmup=10)
    if HAS_TORCH:
        pa, pb = torch.from_numpy(a), torch.from_numpy(b)
        pt_time = timeit(lambda: pa @ pb, runs=20, warmup=10)
    else:
        pt_time = None
    return tt_time, pt_time


def bench_conv_forward():
    # свёртка с приличным батчем и числом каналов
    x = np.random.randn(64, 16, 32, 32).astype(np.float32)
    conv_tt = Conv2d(16, 32, 3, padding=1)
    tx = Tensor(x)
    tt_time = timeit(lambda: conv_tt(tx), runs=20, warmup=10)
    if HAS_TORCH:
        conv_pt = tnn.Conv2d(16, 32, 3, padding=1)
        px = torch.from_numpy(x)
        with torch.no_grad():
            pt_time = timeit(lambda: conv_pt(px), runs=20, warmup=10)
    else:
        pt_time = None
    return tt_time, pt_time


def bench_train_step():
    # полный шаг обучения LeNet: forward+backward+update. батч побольше
    x = np.random.randn(128, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 128)

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
    tt_time = timeit(tt_step, runs=15, warmup=8)

    if HAS_TORCH:
        pm = tnn.Sequential(
            tnn.Conv2d(1, 6, 5, padding=2), tnn.ReLU(), tnn.MaxPool2d(2),
            tnn.Conv2d(6, 16, 5), tnn.ReLU(), tnn.MaxPool2d(2),
            tnn.Flatten(), tnn.Linear(16 * 5 * 5, 10),
        )
        popt = torch.optim.AdamW(pm.parameters(), lr=1e-3)
        plossf = tnn.CrossEntropyLoss()
        px = torch.from_numpy(x)
        py = torch.from_numpy(y).long()

        def pt_step():
            pred = pm(px)
            loss = plossf(pred, py)
            popt.zero_grad()
            loss.backward()
            popt.step()
        pt_time = timeit(pt_step, runs=15, warmup=8)
    else:
        pt_time = None
    return tt_time, pt_time


def show(name, tt_ms, pt_ms):
    if pt_ms is not None:
        ratio = tt_ms / pt_ms
        if ratio >= 1:
            rel = f"медленнее в {ratio:.1f}x"
        else:
            rel = f"быстрее в {1/ratio:.1f}x"
        print(f"{name:22} tinytensor={tt_ms:9.2f}мс  pytorch={pt_ms:9.2f}мс  {rel}")
    else:
        print(f"{name:22} tinytensor={tt_ms:9.2f}мс  (torch недоступен)")


print("=" * 74)
print("Сравнение скорости: tinytensor vs pytorch (CPU)")
print("размеры подобраны так, чтобы доминировал реальный счёт, а не оверхед")
print("=" * 74)

show("matmul 2048x2048", *bench_matmul())
show("conv2d forward", *bench_conv_forward())
show("train step (LeNet)", *bench_train_step())

print("=" * 74)
print("matmul: оба через оптимизированный BLAS/MKL, разрыв небольшой.")
print("conv/train: torch впереди за счёт MKL-DNN ядер, которых нет в наивной")
print("NumPy-реализации. отставание в несколько раз - ожидаемо и честно.")
print("на маленьких размерах цифры врут из-за оверхода - не смотри на них.")
