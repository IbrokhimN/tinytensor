# tinytensor

Маленький самописный autograd на numpy. Тензоры, backprop, пара слоев,
лоссы, оптимизаторы и даталоадер. По сути свой мини-pytorch, только без
плюшек и без cuda (пока).

Никакой производительности тут не ищите, тут просто видно как все
устроено внутри, без магии.

## Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Как работает autograd](#как-работает-autograd)
- [API](#api)
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

Зависимость одна - numpy.

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
- `_backward` - функция которая знает как раскидать градиент на родителей.

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
| `a @ b` | matmul | `dL/da += dL/dz @ bᵀ`, `dL/db += aᵀ @ dL/dz` |
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
активации `relu / leaky_relu / sigmoid / tanh / gelu`, ну и `.backward()`.

### nn

- `Module` - от него наследуются все слои, `forward / parameters() / zero_grad()`, по духу как [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html);
- `Linear(in_features, out_features)` - обычный `y = xW + b`, веса инициализируются по [He/Kaiming init](https://arxiv.org/abs/1502.01852) (`std = sqrt(2/in_features)`);
- `ReLU, LeReLU, Sigmod, Tanh, GELU` - тонкие обертки над методами Tensor;
- `MSELoss` - `mean((pred - target)^2)`, банальщина.

### optim

- `SGD(params, lr, momentum=0.0)` - обычный градиентный спуск, можно с моментом;
- `AdamW(params, lr, betas, eps, weight_decay)` - Adam с отдельным weight decay, см. [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) (в отличие от обычного [Adam](https://arxiv.org/abs/1412.6980), decay тут не лезет в градиент, а сразу режет веса).

### data

- `Dataset / TensorDataset` - обертка над (x, y);
- `DataLoader` - бьет на батчи, можно с shuffle, мини-версия [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html).

## Примеры

- [`examples/01_linear_regression.py`](examples/01_linear_regression.py) - линейная регрессия, Linear + MSELoss + SGD на `y = 3x + 2 + шум`.
- [`examples/02_mnist_mlp.py`](examples/02_mnist_mlp.py) - MLP (Linear -> ReLU -> Linear) с AdamW и DataLoader на синтетике в формате mnist (784 фичи, 10 классов). Настоящего mnist и загрузчиков датасетов тут нет, лень было тащить.

Как выглядит запуск вживую:

```
$ python3 01_linear_regression.py
epoch   0 | loss 30.8016
epoch 180 | loss 0.2365
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

47 штук, гоняют арифметику тензоров и broadcasting, autograd (накопление
градиента, топология, повторный backward), лоссы, оптимизаторы и
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
│   ├── backends/     # заготовка под cuda, пока пусто
│   └── config.py     # сид рандома
├── examples/
├── tests/
├── setup.py
└── requirements.txt
```

## Чего нет

- Только numpy-backend, cuda_gpu.py пустой файл-заглушка.
- Нет кросс-энтропии, сверток, рекуррентных слоев, сохранения модели (см. ToDo в исходном плане проекта).
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