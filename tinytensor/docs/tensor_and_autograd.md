# Tensor и autograd

## Что внутри Tensor

У каждого `Tensor` есть:

- `data` - сами значения (numpy-массив, всегда float32);
- `grad` - градиент, пока не посчитан - None;
- `_prev` - от каких тензоров он произошел (родители в графе);
- `_backward` - функция которая знает как раскидать градиент на родителей;
- `device` - просто метка `"cpu"` или `"cuda"`, влияет только на то, какой matmul дернется (см. [cuda.md](cuda.md)).

```python
from tinytensor.core.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
```

`requires_grad=False` по умолчанию - если тензор не участвует в обучении (например входные данные или маска в dropout), градиент по нему считать не надо, и autograd честно его пропускает.

## Как работает backward

Когда считаете `z = f(x, y)`, по цепному правилу:

```
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
```

`backward()` строит топологический порядок графа (`tinytensor/core/autograd.py`), ставит корню градиент = 1 и идет в обратном порядке, вызывая `_backward()` у каждого узла. Ровно так же устроен micrograd Карпатова, только у него даже покомпактнее:

- [Karpathy - micrograd](https://github.com/karpathy/micrograd) - реализация того же самого на ~100 строк, must see
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) - если нужно разложить backprop по полочкам
- [colah - Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)

Видос по теме, если хочется увидеть как это пишется с нуля вживую: [Karpathy - "The spelled-out intro to neural networks and backpropagation"](https://www.youtube.com/watch?v=VMj-3S1tku0).

Пример руками:

```python
a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a * b          # тензор запоминает откуда он взялся
d = c.sum()
d.backward()        # градиент = 1 в d, потом раскидывается назад
print(a.grad, b.grad)  # [3.] [2.]
```

Важный момент: градиенты **накапливаются**, а не перезаписываются. Поэтому перед каждым новым шагом обучения нужно звать `optimizer.zero_grad()` (или `model.zero_grad()`), иначе градиенты с прошлого шага останутся и посчитаются заново поверх старых.

## Формулы, которые реализованы в Tensor

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

## Broadcasting

Про broadcasting и почему градиент иногда надо досуммировать обратно (`_unbroadcast`) норм объясняют [правила broadcasting в numpy](https://numpy.org/doc/stable/user/basics.broadcasting.html). Смысл в двух словах: если складываете `(2,3)` и `(1,3)`, numpy сам растягивает второй тензор до `(2,3)` для прямого прохода, а вот градиент на обратном надо сжать обратно до `(1,3)` - иначе размерности не сойдутся. Этим и занимается `_unbroadcast`.

## Известное ограничение

`backward()` не кэширует граф между вызовами, каждый раз строит топологию заново, как и в micrograd - никакой лени в духе pytorch (`retain_graph`, `detach` и тд) тут нет.
