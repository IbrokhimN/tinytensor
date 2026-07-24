# tinytensor

Маленький самописный autograd на numpy. Тензоры, backprop, пара слоев,
лоссы, оптимизаторы, даталоадер, плюс теперь есть опциональный cuda-бэкенд
для matmul через cublas. По сути свой мини-pytorch, только совсем без плюшек.

Никакой производительности тут не ищите (ну кроме куды, если она у вас
собралась), тут просто видно как все устроено внутри

## Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Как работает autograd](#как-работает-autograd)
- [API](#api)
- [Cuda-бэкенд](#cuda-бэкенд)
- [Utils](#utils)
- [Сохранение модели](#сохранение-модели)
- [Примеры](#примеры)
- [Тесты](#тесты)
- [Структура](#структура)
- [Чего нет](#чего-нет)
- [Ссылки](#ссылки)

## Установка

```bash
git clone https://github.com/IbrokhimN/tinytensor
cd tinytensor
pip install -e .          # обычная установка
pip install -e .[dev]     # плюс pytest, если хотите гонять тесты
```

Или руками:

```bash
pip install -r requirements.txt
```

Зависимости: numpy и pybind11. Куда собирается сама, если найдется nvcc
(см. ниже), если нет - просто соберется чистый cpu-вариант, ничего
руками включать не надо.

## Быстрый старт

```python
from tinytensor.core.tensor import Tensor
from tinytensor.nn.linear import Linear
from tinytensor.nn.losses import MSELoss
from tinytensor.optim import SGD

model = Linear(in_features=1, out_features=1)
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

x = Tensor([[1.0], [2.0], [3.0]])
y = Tensor([[3.0], [5.0], [7.0]])   # y = 2x + 1

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

print(model.weight.data, model.bias.data)  # ~2.0, ~1.0
```

Рабочие примеры целиком лежат в [`examples/`](examples/).

## Как работает autograd

У каждого `Tensor` есть:

- `data` - сами значения (numpy-массив, всегда float32);
- `grad` - градиент, пока не посчитан - None;
- `_prev` - от каких тензоров он произошел (родители в графе);
- `_backward` - функция которая знает как раскидать градиент на родителей;
- `device` - просто метка `"cpu"` или `"cuda"`, влияет только на то, какой matmul дернется.

Когда считаете `z = f(x, y)`, по цепному правилу:

```
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
```

`backward()` строит топологический порядок графа (`tinytensor/core/autograd.py`),
ставит корню градиент = 1 и идет в обратном порядке, вызывая `_backward()`
у каждого узла. Ровно так же устроен micrograd Карпатова, только у него
даже покомпактнее:

- [Karpathy - micrograd](https://github.com/karpathy/micrograd) - реализация того же самого на ~100 строк, must see
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) - если нужно разложить backprop по полочкам
- [colah - Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)

Если совсем в тему - есть видос Карпатова где он с нуля пишет micrograd и
объясняет каждую строчку: ["The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0).
Собственно тут все то же самое, только на numpy и чуть пошире.

### Формулы, которые реализованы в Tensor

| операция | вперед | производная |
|---|---|---|
| `a + b` | `a + b` | `dL/da += dL/dz`, `dL/db += dL/dz` (плюс схлопывание по broadcast-осям) |
| `a * b` | `a * b` | `dL/da += dL/dz * b`, `dL/db += dL/dz * a` |
| `a @ b` | matmul (cpu или cublas) | `dL/da += dL/dz @ bᵀ`, `dL/db += aᵀ @ dL/dz` |
| `a ** p` | `aᵖ` | `dL/da += dL/dz * p * a^(p-1)` |
| ReLU | `max(0, x)` | 1 при x>0, иначе 0 |
| LeakyReLU | x или αx | 1 при x>0, иначе α |
| sigmoid | `1/(1+e⁻ˣ)` | `σ(x)*(1-σ(x))` |
| tanh | `tanh(x)` | `1 - tanh²(x)` |
| GELU | `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` | см. [статью по GELU](https://arxiv.org/abs/1606.08415), формула там не самая короткая |

Про broadcasting и почему градиент иногда надо досуммировать обратно
(`_unbroadcast`) норм объясняют [правила broadcasting в numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html).

## API

### Tensor

`tinytensor.core.tensor.Tensor` - главный класс. Есть `+ - * @ **`, `.sum()`,
активации `relu / leaky_relu / sigmoid / tanh / gelu`, `.backward()`, и
`.to(device)` чтоб пометить тензор как `"cpu"` или `"cuda"`.

### nn

- `Module` - от него наследуются все слои, `forward / parameters() / zero_grad() / state_dict() / load_state_dict() / save() / load()`, по духу как [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html);
- `Linear(in_features, out_features)` - обычный `y = xW + b`, веса инициализируются по [He/Kaiming init](https://arxiv.org/abs/1502.01852) (`std = sqrt(2/in_features)`);
- `ReLU, LeReLU, Sigmod, Tanh, GELU` - тонкие обертки над методами Tensor;
- `MSELoss` - `mean((pred - target)^2)`, банальщина.

### optim

- `SGD(params, lr, momentum=0.0)` - обычный градиентный спуск, можно с моментом;
- `AdamW(params, lr, betas, eps, weight_decay)` - Adam с отдельным weight decay, см. [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) (в отличие от обычного [Adam](https://arxiv.org/abs/1412.6980), decay тут не лезет в градиент, а сразу режет веса).

### data

- `Dataset / TensorDataset` - обертка над (x, y);
- `DataLoader` - бьет на батчи, можно с shuffle, мини-версия [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html).

## Cuda-бэкенд

При установке `setup.py` сам проверяет есть ли `nvcc`, и если да - собирает
`tinytensor.cuda_ops`, маленький pybind11-модуль, который дергает
`cublasSgemm` (`tinytensor/backends/cuda_gpu.cu` + `cuda_binding.cpp`).
Если куды нет, сборка просто идет по cpu-пути, ничего не ломается.

Сейчас куда греет только `matmul` - если у любого из двух тензоров
`device == "cuda"` и модуль собрался, `Tensor.__matmul__` уходит в
`cuda_ops.matmul` вместо `numpy`. Все остальные операции (сложение,
активации и тд) все равно считаются на cpu через numpy, реального
управления памятью на gpu тут нет - данные каждый раз гоняются
host <-> device внутри одного вызова.

Что почитать если охота лучше понять что там происходит:

- [cuBLAS docs (cublasSgemm)](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
- [pybind11 docs](https://pybind11.readthedocs.io/en/stable/) - как из C++/CUDA дергать функции из питона

## Utils

`tinytensor.utils`:

- `progress_bar` / `train_bar` (алиас) - генератор-обертка, рисует прогрессбар в консоли:
  ```python
  from tinytensor.utils import train_bar
  for epoch in train_bar(range(100), prefix="обучение"):
      ...
  ```
- `EarlyStopping(model, patience=5, min_delta=0.0, restore_best_weights=True)` - следит за val_loss, останавливает обучение и откатывает модель на лучший чекпоинт если она перестала улучшаться:
  ```python
  from tinytensor.utils import EarlyStopping
  early_stopping = EarlyStopping(model, patience=7)
  for epoch in range(100):
      val_loss = evaluate(model)
      if early_stopping(val_loss):
          break
  ```

## Сохранение модели

У `Module` есть `save(filepath)` / `load(filepath)` (обычный pickle поверх
`state_dict()`), формат файла - `.tt`. По задумке должно работать вот так:

```python
model.save("model.tt")
model.load("model.tt")
```

На практике сейчас тут баг: `state_dict()` не делает `return`, а `load()`
открывает файл в режиме `"wb"` вместо `"rb"` - то есть load сам затирает
файл перед тем как его же читать. Так что прямо сейчас save/load реально
не работают, несмотря на то что в примере `01_linear_regression.py` вызов
`model.save(...)` есть. README не трогает код, так что просто имейте это
в виду пока кто-нибудь не поправит эти две строчки.

## Примеры

- [`examples/01_linear_regression.py`](examples/01_linear_regression.py) - линейная регрессия, Linear + MSELoss + SGD на синтетике, с прогрессбаром (`progress_bar`) и попыткой сохранить модель в `test_model.tt`.
- [`examples/02_mnist_mlp.py`](examples/02_mnist_mlp.py) - MLP (Linear -> ReLU -> Linear) с AdamW и DataLoader на синтетике в формате mnist (784 фичи, 10 классов). Настоящего mnist и загрузчиков датасетов тут нет, лень было тащить.

Как выглядит запуск вживую:

```
$ python3 01_linear_regression.py
обучение |██████████████████████████████| 100.0% [0.0s]
выученные параметры: weight ~ 2.995, bias ~ 1.996

$ python3 02_mnist_mlp.py
epoch 1/5 | avg loss 2.0998
epoch 5/5 | avg loss 0.0481
```

## Тесты

```bash
pip install -e .[dev]
python3 -m pytest tests/ -v
```

Гоняют арифметику тензоров и broadcasting, autograd (накопление
градиента, топология, повторный backward), лоссы, оптимизаторы, слои nn и
Dataset/DataLoader.

> Важно: запускать через pytest из корня репы (или после `pip install -e .`),
> а не `python3 test_x.py` из папки tests - иначе tinytensor просто не найдется.

## Структура

```
tinytensor/
├── tinytensor/
│   ├── core/        # Tensor, autograd, ops
│   ├── nn/          # Module, Linear, активации, MSELoss
│   ├── optim/        # SGD, AdamW
│   ├── data/         # Dataset, DataLoader
│   ├── backends/     # cpu_numpy.py (заглушка) + cuda_gpu.cu / cuda_binding.cpp (cublas)
│   ├── utils/        # progress_bar, train_bar, EarlyStopping
│   └── config.py     # сид рандома
├── examples/
├── tests/
├── setup.py
└── requirements.txt
```

## Чего нет

- Нет кросс-энтропии, сверток, рекуррентных слоев
- Cuda-бэкенд ускоряет только matmul, остальное все равно на cpu, и никакого реального управления gpu-памятью между операциями нет.
- save/load модели сейчас сломаны (см. раздел [Сохранение модели](#сохранение-модели)).
- backward не кэширует граф между вызовами, каждый раз строит топологию заново, как и в micrograd - никакой лени в духе pytorch тут нет.

## Ссылки

- [Andrej Karpathy - micrograd (GitHub)](https://github.com/karpathy/micrograd) - основной референс для всего autograd
- [Andrej Karpathy - видео про backprop и micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [CS231n - Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- [colah - Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
- [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [He et al. - инициализация весов](https://arxiv.org/abs/1502.01852)
- [Kingma & Ba - Adam](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter - AdamW](https://arxiv.org/abs/1711.05101)
- [Hendrycks & Gimpel - GELU](https://arxiv.org/abs/1606.08415)
- [PyTorch - torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch - torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html)
- [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html)
- [pybind11 docs](https://pybind11.readthedocs.io/en/stable/)
