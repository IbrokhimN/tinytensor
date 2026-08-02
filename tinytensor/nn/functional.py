import numpy as np

from tinytensor.core.tensor import get_array_module

# вцелом идея im2col это то что изначально интуитивный метод перебора окна по изображению является просто написать 6 циклов для сдвигов и тд однако это ооочень медленный метод, и поэтому мы преобразовываем в 1 матрицу и умножаем и это все будет в разыыы быстрее чем наивный метод перебора

def im2col_indices(x, kh, kw, padding=1, stride=1):
    xp = get_array_module(x)          # np или cupy, смотря где данные
    p = padding
    #делаем паддинг
    x_padded = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    N, C, H, W = x.shape
    out_h = int((H + 2 * p - kh) / stride + 1)
    out_w = int((W + 2 * p - kw) / stride + 1)

    i0 = xp.repeat(xp.arange(kh), kw)
    i0 = xp.tile(i0, C)
    i1 = stride * xp.repeat(xp.arange(out_h), out_w)
    j0 = xp.tile(xp.arange(kw), kh * C)
    j1 = stride * xp.tile(xp.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), kh * kw).reshape(-1, 1)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * C, -1)
    return cols, out_h, out_w

def col2_im_indices(cols, x_shape, kh, kw, padding=1, stride=1):
    xp = get_array_module(cols)       # np или cupy, смотря где данные
    N, C, H, W = x_shape
    p = padding
    H_padded, W_padded = H + 2 * p, W + 2 * p
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    out_h = int((H + 2 * p - kh) / stride + 1)
    out_w = int((W + 2 * p - kw) / stride + 1)

    i0 = xp.repeat(xp.arange(kh), kw)
    i0 = xp.tile(i0, C)
    i1 = stride * xp.repeat(xp.arange(out_h), out_w)
    j0 = xp.tile(xp.arange(kw), kh * C)
    j1 = stride * xp.tile(xp.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), kh * kw).reshape(-1, 1)

    cols_reshaped = cols.reshape(C * kh * kw, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # scatter-add: у numpy это np.add.at, у cupy - cupyx.scatter_add
    if xp is np:
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    else:
        import cupyx
        cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)

    if p == 0:
        return x_padded
    return x_padded[:, :, p:-p, p:-p]
