import time
import tracemalloc
import numpy as np


def count_params(model):
    # общее число обучаемых параметров модели
    total = 0
    for p in model.parameters():
        total += int(np.prod(p.data.shape))
    return total


def benchmark(model, x, runs=100, warmup=10, batch_size=None):
    # меряет latency и пиковую память инференса.
    # x входные данные (numpy), runs число замеров, warmup прогрев
    # возвращает dict с median/mean latency (мс), peak RAM (МБ), числом параметров
    model.eval()

    # первые прогоны часто медленнее
    for _ in range(warmup):
        if batch_size is not None:
            model.predict(x, batch_size=batch_size)
        else:
            _run_once(model, x)

    # замер времени
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        if batch_size is not None:
            model.predict(x, batch_size=batch_size)
        else:
            _run_once(model, x)
        times.append((time.perf_counter() - t0) * 1000.0)   # в мс

    times = np.array(times)

    # пиковая память одного прогона
    tracemalloc.start()
    if batch_size is not None:
        model.predict(x, batch_size=batch_size)
    else:
        _run_once(model, x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "latency_median_ms": float(np.median(times)),
        "latency_mean_ms": float(times.mean()),
        "latency_std_ms": float(times.std()),
        "peak_ram_mb": peak / (1024 ** 2),
        "params": count_params(model),
        "runs": runs,
    }

    print(f"latency: median={result['latency_median_ms']:.2f} мс  "
          f"mean={result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} мс")
    print(f"peak RAM: {result['peak_ram_mb']:.2f} МБ")
    print(f"параметры: {result['params']:,}")
    return result


def _run_once(model, x):
    from tinytensor.core.tensor import Tensor
    xb = x if isinstance(x, Tensor) else Tensor(x)
    return model(xb)


def model_size(model, filepath=None):
    import os
    if filepath is not None:
        return os.path.getsize(filepath) / 1024.0
    # временный файл
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".tt", delete=False)
    tmp.close()
    try:
        model.save(tmp.name)
        kb = os.path.getsize(tmp.name) / 1024.0
    finally:
        os.remove(tmp.name)
    return kb


def flops(model, input_shape):
    from tinytensor.core.tensor import Tensor
    from tinytensor.nn.conv import Conv2d
    from tinytensor.nn.linear import Linear

    layers = getattr(model, "layers", None)
    if layers is None:
        layers = [a for a in model.__dict__.values() if hasattr(a, "layers")]
        layers = layers[0].layers if layers else []

    x = Tensor(np.zeros(input_shape, dtype=np.float32))
    total = 0
    for layer in layers:
        if isinstance(layer, Conv2d):
            x = layer(x)
            # macs = out_h * out_w * out_ch * in_ch * kh * kw
            _, out_ch, out_h, out_w = x.data.shape
            total += out_h * out_w * out_ch * layer.in_channels * layer.kh * layer.kw
        elif isinstance(layer, Linear):
            # macs = in_features * out_features
            in_f, out_f = layer.weight.data.shape
            total += in_f * out_f
            x = layer(x)
        else:
            x = layer(x)
    return total
