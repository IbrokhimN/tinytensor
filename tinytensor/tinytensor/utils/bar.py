import time
import sys


def progress_bar(iterable, prefix="", length=30):
    total = len(iterable)
    start_time = time.time()
    # рисуем бар только в настоящий терминал, иначе (файл/пайп) молчим про перерисовку
    is_tty = sys.stdout.isatty()
    for i, item in enumerate(iterable):
        yield item
        if not is_tty:
            continue
        percent = (i + 1) / total
        filled = int(length * percent)
        bar = "█" * filled + "-" * (length - filled)
        elapsed = time.time() - start_time
        # \r в начало строки, \033[K стирает старый хвост -> строка не множится
        sys.stdout.write(f"\r\033[K{prefix} |{bar}| {percent*100:.1f}% [{elapsed:.1f}s]")
        sys.stdout.flush()
    if is_tty:
        sys.stdout.write("\n")
        sys.stdout.flush()


#алиас для тех кто привык train_bar писать
train_bar = progress_bar


"""
пользоваться надо крч вот так:
for epoch in train_bar(range(100), prefix="обучение"):
    ...
"""
