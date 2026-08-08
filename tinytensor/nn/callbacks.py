import numpy as np


class Callback:
    def on_train_begin(self, model):
        pass

    def on_epoch_end(self, epoch, logs, model):
        return False

    def on_train_end(self, model):
        pass


class EarlyStop(Callback):
    # стоп если метрика не улучшается patience эпох подряд
    def __init__(self, monitor="val_loss", patience=5, mode="min"):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs, model):
        val = logs.get(self.monitor)
        if val is None:
            return False
        # улучшилось или нет в зависимости от mode
        better = (self.best is None or
                  (val < self.best if self.mode == "min" else val > self.best))
        if better:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"ранняя остановка (нет улучшения {self.monitor} за {self.patience} эпох)")
                return True
        return False


class ModelCheckpoint(Callback):
    # сохраняет модель когда monitor улучшается
    def __init__(self, filepath, monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def on_epoch_end(self, epoch, logs, model):
        val = logs.get(self.monitor)
        if val is None:
            return False
        better = (self.best is None or
                  (val < self.best if self.mode == "min" else val > self.best))
        if better:
            self.best = val
            model.save(self.filepath)
            print(f"  чекпоинт сохранён (эпоха {epoch+1}, {self.monitor}={val:.4f})")
        return False


class LRScheduler(Callback):
    # дёргает scheduler.step() в конце эпохи
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs, model):
        self.scheduler.step()
        return False


class CSVLogger(Callback):
    # пишет метрики каждой эпохи в csv
    def __init__(self, filepath):
        self.filepath = filepath
        self.keys = None
        self.f = None

    def on_train_begin(self, model):
        self.f = open(self.filepath, "w")

    def on_epoch_end(self, epoch, logs, model):
        if self.keys is None:
            self.keys = list(logs.keys())
            self.f.write("epoch," + ",".join(self.keys) + "\n")
        row = [str(epoch + 1)] + [f"{logs[k]:.6f}" for k in self.keys]
        self.f.write(",".join(row) + "\n")
        self.f.flush()
        return False

    def on_train_end(self, model):
        if self.f is not None:
            self.f.close()
