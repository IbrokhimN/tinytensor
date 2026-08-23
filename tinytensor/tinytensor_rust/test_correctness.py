# проверка что rust-версия im2col даёт тот же результат что numpy.
# запускать ПОСЛЕ сборки (maturin develop --release).
import numpy as np

try:
    import tinytensor_rust
except ImportError:
    print("tinytensor_rust не собран. сначала: maturin develop --release")
    raise SystemExit

# numpy-эталон (независимая реализация для сверки)
def im2col_numpy_ref(x, kh, kw, padding, stride):
    N, C, H, W = x.shape
    p = padding
    if p > 0:
        xp = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=np.float32)
        xp[:, :, p:p + H, p:p + W] = x
    else:
        xp = x
    out_h = (H + 2 * p - kh) // stride + 1
    out_w = (W + 2 * p - kw) // stride + 1
    i0 = np.repeat(np.arange(kh), kw); i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * C); j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)
    cols = xp[:, k, i, j].transpose(1, 2, 0).reshape(kh * kw * C, -1)
    return cols

np.random.seed(0)
configs = [
    (2, 3, 8, 8, 3, 3, 1, 1),
    (4, 1, 28, 28, 5, 5, 0, 1),
    (2, 4, 10, 10, 3, 3, 1, 1),
    (1, 2, 6, 6, 3, 3, 0, 2),
    (3, 2, 12, 12, 3, 3, 1, 2),
    (5, 1, 28, 28, 5, 5, 2, 1),
]

all_ok = True
for (N, C, H, W, kh, kw, p, s) in configs:
    x = np.random.randn(N, C, H, W).astype(np.float32)
    x = np.ascontiguousarray(x)
    rust = tinytensor_rust.im2col(x, kh, kw, p, s)
    ref = im2col_numpy_ref(x, kh, kw, p, s)
    ok = np.allclose(rust, ref, atol=1e-5)
    all_ok = all_ok and ok
    print(f"N={N} C={C} H={H} k={kh} p={p} s={s}: {'OK' if ok else 'FAIL <<<'}")

print("\nВСЁ СОВПАЛО - rust корректен" if all_ok else "\nЕСТЬ РАСХОЖДЕНИЯ")
