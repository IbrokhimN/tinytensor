import numpy as np

from tinytensor.core.tensor import get_array_module

# пробуем подключить rust-ускорение im2col.
# если модуль не собран - работаем на чистом numpy (fallback).
try:
    import tinytensor_rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# вцелом идея im2col это то что изначально интуитивный метод перебора окна по изображению является просто написать 6 циклов для сдвигов и тд однако это ооочень медленный метод, и поэтому мы преобразовываем в 1 матрицу и умножаем и это все будет в разыыы быстрее чем наивный метод перебора

# при обучении форма батчей постоянна и значит считаем индексы один раз и дальше берём готовые.
_idx_cache = {}


def _get_indices(xp, N, C, H, W, kh, kw, padding, stride, out_h, out_w):
    # ключ кэша это всё что влияет на индексы кроме самих данных
    key = (id(xp), C, kh, kw, padding, stride, out_h, out_w)
    cached = _idx_cache.get(key)
    if cached is not None:
        return cached

    i0 = xp.repeat(xp.arange(kh), kw)
    i0 = xp.tile(i0, C)
    i1 = stride * xp.repeat(xp.arange(out_h), out_w)
    j0 = xp.tile(xp.arange(kw), kh * C)
    j1 = stride * xp.tile(xp.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), kh * kw).reshape(-1, 1)

    _idx_cache[key] = (k, i, j)
    return k, i, j


def im2col_indices(x, kh, kw, padding=1, stride=1):
    xp = get_array_module(x)          # np или cupy, смотря где данные
    p = padding
    N, C, H, W = x.shape

    # rust-ускорение только для cpu (numpy). на cuda остаётся старый путь.
    if _HAS_RUST and xp is np:
        x_f32 = np.ascontiguousarray(x, dtype=np.float32)
        cols = tinytensor_rust.im2col(x_f32, kh, kw, p, stride)
        out_h = int((H + 2 * p - kh) / stride + 1)
        out_w = int((W + 2 * p - kw) / stride + 1)
        return cols, out_h, out_w

    # паддинг через предвыделение быстрее чем xp.pad
    if p > 0:
        x_padded = xp.zeros((N, C, H + 2 * p, W + 2 * p), dtype=x.dtype)
        x_padded[:, :, p:p + H, p:p + W] = x
    else:
        x_padded = x
    out_h = int((H + 2 * p - kh) / stride + 1)
    out_w = int((W + 2 * p - kw) / stride + 1)

    # индексы берём из кэша (или считаем один раз)
    k, i, j = _get_indices(xp, N, C, H, W, kh, kw, padding, stride, out_h, out_w)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * C, -1)
    return cols, out_h, out_w


def col2_im_indices(cols, x_shape, kh, kw, padding=1, stride=1):
    xp = get_array_module(cols)       # np или cupy, смотря где данные
    N, C, H, W = x_shape
    p = padding
    H_padded, W_padded = H + 2 * p, W + 2 * p

    out_h = int((H + 2 * p - kh) / stride + 1)
    out_w = int((W + 2 * p - kw) / stride + 1)

    # те же кэшированные индексы что и в im2col
    k, i, j = _get_indices(xp, N, C, H, W, kh, kw, padding, stride, out_h, out_w)

    cols_reshaped = cols.reshape(C * kh * kw, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    if xp is np:
        # np.add.at медленный. считаем через плоские индексы и bincount -
        # это в разы быстрее для накопления с повторяющимися индексами.
        lin = (k * H_padded + i) * W_padded + j
        lin = xp.broadcast_to(lin, cols_reshaped.shape[1:]).ravel()
        plane = C * H_padded * W_padded
        x_padded = xp.zeros((N, plane), dtype=cols.dtype)
        for n in range(N):
            x_padded[n] = xp.bincount(lin, weights=cols_reshaped[n].ravel(),
                                      minlength=plane)
        x_padded = x_padded.reshape(N, C, H_padded, W_padded)
    else:
        x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        import cupyx
        cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)

    if p == 0:
        return x_padded
    return x_padded[:, :, p:-p, p:-p]
