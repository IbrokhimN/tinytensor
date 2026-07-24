# Сохранение модели

## Как пользоваться

```python
model.save("model.tt")
model.load("model.tt")
```

Формат файла - `.tt`, внутри обычный pickle поверх `state_dict()`. Ничего особенного, просто словарь `{имя_параметра: numpy-массив}`.

## Как это устроено внутри

```python
def state_dict(self):
    sdict = {}
    for name, param in self._get_named_params().items():
        sdict[name] = param.data
    return sdict

def load_state_dict(self, sdict):
    for name, param in self._get_named_params().items():
        if name in sdict:
            param.data = sdict[name]

def save(self, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(self.state_dict(), f)

def load(self, filepath):
    with open(filepath, "rb") as f:
        sd = pickle.load(f)
        self.load_state_dict(sd)
```

`_get_named_params` рекурсивно проходит по всем вложенным `Module` и собирает имена в стиле `fc1.weight`, `fc1.bias`, `fc2.weight` и тд, так что модель с несколькими слоями сохраняется/загружается целиком одним вызовом.

## Грабли, на которые уже наступали

Открывать файл в правильном режиме - тут легко перепутать местами:

- `save` пишет в файл -> нужен режим `"wb"` (write binary)
- `load` читает из файла -> нужен режим `"rb"` (read binary)

Если перепутать (`save` в `"rb"` или `load` в `"wb"`) - либо упадет сразу с `io.UnsupportedOperation`, либо (что хуже) `load()` в режиме `"wb"` тихо затрет файл перед чтением, и потом `pickle.load` упадет на пустом файле. Мнемоника простая: save = **w**rite = "wb", load = **r**ead = "rb".

Второй баг, который тоже словили - `state_dict()` без `return` в конце. Функция честно собирает словарь `sdict`, но если забыть `return sdict` - метод всегда отдаст `None`, и `save()` тихо запишет в файл `None` вместо весов, без единой ошибки. Проверяйте после правок руками:

```python
before = model.weight.data.copy()
model.save("test.tt")
model2 = SameArchitecture()
model2.load("test.tt")
assert (before == model2.weight.data).all()
```

## Ограничения

- Оптимизатор (`m`/`v`-буферы AdamW, `velocities` у SGD) в `.tt` не сохраняются, только веса модели. Если продолжаете обучение после загрузки - оптимизатор стартует с чистых буферов.
- Формат не versioned - если поменяете архитектуру модели (добавите/уберете слой) и попробуете загрузить старый `.tt`, часть весов из `state_dict` просто не найдет соответствия и будет молча проигнорирована (`if name in sdict`), без предупреждения. Так что храните рядом с чекпоинтом информацию о версии архитектуры сами, если это важно.
