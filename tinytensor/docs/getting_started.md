# Getting Started

## Установка

```bash
pip install pytinytensor
```

Или из исходников:

```bash
git clone https://github.com/IbrokhimN/tinytensor
cd tinytensor
pip install -e .          # обычная установка
pip install -e .[dev]     # плюс pytest, если хотите гонять тесты
```

Зависимости: numpy и pybind11. Куда собирается сама, если найдется nvcc и рядом реально лежат `libcudart`/`libcublas` (см. [cuda.md](cuda.md)), если нет - тихо соберется чистый cpu-вариант, ничего руками включать не надо.

## Быстрый старт

Обычная линейная регрессия, чтоб пощупать основные кирпичи разом: `Tensor`, `Linear`, `MSELoss`, `SGD`.

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

Стандартный цикл обучения тут ровно такой же как в pytorch: `zero_grad -> forward -> loss -> backward -> step`. Никакой магии, автограда, спрятанного за десятью слоями абстракций, тут нет - можно прямо влезть в код и посмотреть что происходит на каждом шаге (см. [tensor_and_autograd.md](tensor_and_autograd.md)).

## Что дальше почитать

- [tensor_and_autograd.md](tensor_and_autograd.md) - как устроен Tensor и backward под капотом
- [nn.md](nn.md) - слои, активации, dropout, лоссы
- [optim.md](optim.md) - SGD и AdamW
- [data.md](data.md) - Dataset и DataLoader
- [utils.md](utils.md) - progress_bar и EarlyStopping
- [model_saving.md](model_saving.md) - save/load модели
- [cuda.md](cuda.md) - опциональный gpu-бэкенд
- [faq.md](faq.md) - частые грабли, на которые уже наступили

Рабочие example-скрипты целиком лежат в [`examples/`](../examples/).
