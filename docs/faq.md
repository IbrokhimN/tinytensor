# FAQ / частые грабли

Сюда скидываю все, на что уже наступали в процессе разработки, чтобы второй раз не наступать.

## `ModuleNotFoundError: No module named 'pybind11'` при `python3 -m build`

`setup.py` делает `import pybind11` наверху файла, а pip при сборке создает **изолированное окружение**, куда по умолчанию ставится только `setuptools`. Нужен `pyproject.toml` рядом с `setup.py`:

```toml
[build-system]
requires = ["setuptools>=61", "wheel", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"
```

Без этого файла (или если он потерялся при переносе на новую машину/ОС) сборка будет падать с этой ошибкой каждый раз.

## `pip install pytinytensor` падает с `cannot find -lcudart` / `-lcublas`

`nvcc` нашелся, а сами библиотеки `libcudart`/`libcublas` физически не установлены (частый случай с conda, где `nvcc` ставится отдельно от рантайм-либ). См. [cuda.md](cuda.md) - в актуальной версии `setup.py` эта ситуация уже не роняет установку, а откатывается на cpu. Если у вас более старая версия и установка падает целиком - обновите `setup.py`.

## `save`/`load` модели не работают

Классика - перепутанные режимы открытия файла. `save` = запись = `"wb"`, `load` = чтение = `"rb"`. Подробнее и с примером проверки в [model_saving.md](model_saving.md).

## `Dropout`/кастомный слой ругается `AttributeError: 'X' object has no attribute 'training'`

Забыли вызвать `super().__init__()` в конструкторе своего слоя. `self.training` появляется только через `Module.__init__`. Сами один раз забыли это сделать в `LeReLU` - ловите на будущее.

## Модель на инференсе ведет себя как на трейне (dropout продолжает вырубать нейроны)

Не забыли позвать `model.eval()` перед инференсом? `train()`/`eval()` рекурсивно проставляют `self.training` по всем вложенным модулям, но если гоняете слои руками мимо основного `forward` - проверьте что флаг реально проставлен на нужном объекте.

## `TypeError: unsupported operand type(s) for *: 'float' and 'memoryview'` в оптимизаторе

Был баг в `optim/optimizer.py` - код пытался сделать `p.grad.data`, думая что распаковывает `Tensor`, но `p.grad` уже обычный `np.ndarray`, а у ndarray `.data` это `memoryview`, а не значения. Уже пофикшено в текущей версии, но если тянете старую копию кода откуда-то - это первое, что стоит проверить, если `optimizer.step()` падает с такой ошибкой.

## В `git` неожиданно вложенная копия `tinytensor/tinytensor/tinytensor/...`

Один раз `setup.py build`/`pip install -e .` был запущен не из корня репозитория, а внутри самого пакета, и результат случайно попал в `git add .`. Проверить у себя:

```bash
git ls-files | grep "tinytensor/tinytensor/"
```

Если что-то нашлось - см. историю коммитов на предмет `cleanup: remove accidentally committed nested package copy` или чистите руками (`git rm -r --cached`, обновить `.gitignore`, закоммитить).

## `twine upload` ругается `Binary wheel ... has an unsupported platform tag 'linux_x86_64'`

PyPI не принимает голые линуксовые wheel'ы, только `manylinux_*`. Проще всего заливать только sdist:

```bash
twine upload dist/pytinytensor-X.Y.Z.tar.gz
```

Подробнее про паблишинг и cuda-wheel в [cuda.md](cuda.md#публикация-wheel-с-кудой).

## KDE/GNOME вылезает с окном про пароль от кошелька при `twine upload`

Это `keyring` пытается сохранить токен в системное хранилище паролей, к делу не относится. Жмите Cancel, `twine` спросит токен обычным текстом в консоли. Если раздражает - `pip uninstall keyring`.

## PyPI пишет "The author of this package has not provided a project description"

Либо не добавлен `long_description` в `setup.py` (нужно читать `README.md` и передавать `long_description_content_type="text/markdown"`), либо посмотрели страницу старой версии, у которой описания и не было - смотрите `pypi.org/project/pytinytensor/X.Y.Z/` с актуальным номером версии.

## Нельзя перезалить ту же версию на PyPI

Да, специально так сделано, даже если удалить релиз. Просто бампайте версию (`0.1.1` -> `0.1.2`) перед каждой новой заливкой.
