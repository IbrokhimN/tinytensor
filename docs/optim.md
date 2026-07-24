# optim

## Базовый цикл

Что SGD, что AdamW - юзаются одинаково:

```python
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

`optimizer.zero_grad()` и `model.zero_grad()` делают одно и то же (обнуляют `.grad` у параметров) - `Optimizer` просто хранит тот же список `parameters()`, что вы ему передали.

## SGD

```python
from tinytensor.optim import SGD

opt = SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
```

Обычный градиентный спуск: `w = w - lr * grad`. С моментом (`momentum > 0`) заводится буфер скорости на каждый параметр:

```
v = momentum * v + grad
w = w - lr * v
```

С `momentum=0` (дефолт) буферы скорости вообще не заводятся, чтобы зря память не жрать.

## AdamW

```python
from tinytensor.optim import AdamW

opt = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

Adam с отдельным weight decay, см. [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101) - в отличие от обычного [Adam](https://arxiv.org/abs/1412.6980), decay тут не лезет в градиент (и соответственно в моменты `m`/`v`), а сразу режет веса напрямую. На практике почти всегда лучше сходится чем ванильный Adam, если у вас вообще есть regularization через weight decay.

Внутри хранит по паре `(m, v)`-буферов на каждый параметр плюс счетчик шагов `t` (нужен для bias-correction на первых итерациях, иначе на старте оптимизатор будет занижать шаги).

## Какой выбрать

Если модель маленькая и хочется просто быстро проверить что вообще что-то учится - берите `SGD` с моментом `0.9`, дешево и просто. Если модель побольше или лосс скачет - `AdamW` почти всегда сходится увереннее и меньше требует тюнинга lr.
