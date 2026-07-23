import time


def progress_bar(iterable, prefix="", length=30):
    total = len(iterable)
    start_time = time.time()
    for i, item in enumerate(iterable):
        yield item  # ИСПРАВЛЕНО: возвращаем элемент, а не модуль time
        percent = (i + 1) / total
        filled = int(length * percent)
        bar = "█" * filled + "-" * (length - filled)
        elapsed = time.time() - start_time

        print(f"\r{prefix} |{bar}| {percent*100:.1f}% [{elapsed:.1f}s]", end="")
    print()


#алиас для тех кто привык train_bar писать
train_bar = progress_bar


"""
пользоваться надо крч вот так:
for epoch in train_bar(range(100), prefix="обучение"):
    ...
"""
