# tinytensor/core/autograd.py
import numpy as np


def backward(target_tensor):
    # эт я вообще сделал чтоб файл не пустовал, но это архитектура micrograd как у Андрея Карпатова,
    # если кто то шарит за нормальный движок autograd, то PRните
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(target_tensor)

    target_tensor.grad = np.ones_like(target_tensor.data, dtype=np.float32)

    for v in reversed(topo):
        v._backward()