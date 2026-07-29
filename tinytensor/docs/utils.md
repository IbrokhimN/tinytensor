# utils

## progress_bar / train_bar

Генератор-обертка, рисует прогрессбар в консоли поверх любого итерируемого:

```python
from tinytensor.utils import train_bar   # алиас для progress_bar, кому как привычнее

for epoch in train_bar(range(100), prefix="обучение"):
    ...
```

Печатает что-то типа:
```
обучение |██████████████████████████████| 100.0% [12.3s]
```

Работает с любым объектом у которого есть `len()`, необязательно с `range` - можно и на `DataLoader` навесить, если хочется видеть прогресс по батчам, а не по эпохам.

## EarlyStopping

Следит за val_loss, останавливает обучение и (по желанию) откатывает модель на лучший чекпоинт, если она перестала улучшаться:

```python
from tinytensor.utils import EarlyStopping

model = MLP()
early_stopping = EarlyStopping(model, patience=7, min_delta=0.01, restore_best_weights=True)

for epoch in range(100):
    train_one_epoch(model)
    val_loss = evaluate(model)

    if early_stopping(val_loss):
        print(f"обучение прекратилось на эпохе {epoch+1}")
        break
```

Как это работает:

- `patience` - сколько эпох подряд можно не улучшаться, прежде чем реально остановиться;
- `min_delta` - минимальное улучшение, которое считается "реальным" (если лосс упал меньше чем на `min_delta` - это не считается прогрессом, счетчик `patience` тикает дальше);
- `restore_best_weights` - если True, при остановке модель откатится на веса с лучшей эпохи (через `state_dict()`/`load_state_dict()` под капотом, деньги на диск не тратятся, все в памяти через `copy.deepcopy`).

`EarlyStopping.__call__` принимает как обычное число, так и `Tensor` (сам вытащит `.data`), так что можно скармливать прямо результат `loss_fn(...)` без ручной распаковки.
