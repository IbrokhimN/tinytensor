# nn

## Module

От него наследуются все слои. Основные штуки:

```python
from tinytensor.nn.modules import Module

class MyLayer(Module):
    def __init__(self):
        super().__init__()   # обязательно, иначе не будет self.training
        ...

    def forward(self, x):
        ...
```

- `forward(*args, **kwargs)` - тут вся логика слоя, надо переопределять в каждом наследнике;
- `__call__` - просто дергает `forward`, поэтому `model(x)` и `model.forward(x)` это одно и то же;
- `parameters()` - собирает все `Tensor` с `requires_grad=True`, рекурсивно проходясь по вложенным модулям;
- `zero_grad()` - обнуляет градиенты у всех параметров;
- `train(mode=True)` / `eval()` - переключает `self.training` рекурсивно по всем вложенным модулям (нужно для `Dropout`, см. ниже);
- `state_dict()` / `load_state_dict()` / `save()` / `load()` - сохранение весов, см. [model_saving.md](model_saving.md).

По духу все это как [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), только без магии типа хуков и буферов.

!!! warning "Не забывайте super().__init__()"
    Если пишете свой слой и в нем не только `Tensor`-параметры, а еще вложенные модули (например `self.fc1 = Linear(...)`) - обязательно зовите `super().__init__()` в конструкторе, иначе `self.training` не появится и `Dropout`/`train()`/`eval()` сломаются. Мы сами один раз забыли это сделать в `LeReLU` и словили `AttributeError`.

## Linear

Обычный полносвязный слой `y = xW + b`:

```python
from tinytensor.nn.linear import Linear

layer = Linear(in_features=784, out_features=128)
out = layer(x)  # x.shape == (batch, 784) -> out.shape == (batch, 128)
```

Веса инициализируются по [He/Kaiming init](https://arxiv.org/abs/1502.01852) (`std = sqrt(2/in_features)`) - неплохой дефолт под ReLU-семейство активаций.

## Активации

Все тонкие обертки над методами `Tensor`, ничего своего не хранят кроме параметра (если он есть):

```python
from tinytensor.nn.activations import ReLU, LeReLU, Sigmod, Tanh, GELU

ReLU()(x)
LeReLU(alpha=0.01)(x)   # да, именно LeReLU, а не LeakyReLU
Sigmod()(x)              # да, именно Sigmod, а не Sigmoid - опечатка так и живет в API
Tanh()(x)
GELU()(x)
```

## Dropout

```python
from tinytensor.nn.dropout import Dropout

drop = Dropout(p=0.5)   # 0.5 = половина нейронов вырубается на трейне
```

Классический inverted dropout: на трейне часть значений зануляется, а оставшиеся домножаются на `1/(1-p)`, чтобы среднее не поехало. На инференсе (`model.eval()`) дропаут ничего не делает и просто пропускает вход как есть.

!!! warning "Без model.eval() дропаут не выключается"
    У `Module` нет отдельного глобального переключателя, `self.training` у каждого модуля свой, и `train()`/`eval()` на верхней модели рекурсивно проставляет флаг всем вложенным слоям. Так что если делаете кастомный форвард-пасс без вызова через `model()`, а руками по слоям - не забудьте прогнать `eval()` перед инференсом.

!!! danger "p=1.0 сломает вам обучение"
    Не используйте `p=1.0` - на инвертированном дропауте это деление на ноль (`1/(1-1)`), получите `NaN` на ровном месте без единой ошибки в консоли.

## MSELoss

```python
from tinytensor.nn.losses import MSELoss

loss_fn = MSELoss()
loss = loss_fn(pred, target)   # mean((pred - target)^2)
```

Кросс-энтропии пока нет, только MSE. Для классификации можно гонять через one-hot + MSE как временное решение (см. `examples/02_mnist_mlp.py`), но это не то же самое что нормальный log-loss - если у вас реальная классификационная задача, качество будет хуже чем с кросс-энтропией.
