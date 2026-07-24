import copy


class EarlyStopping:
    def __init__(self, model, patience=5, min_delta=0.0, restore_best_weights=True):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss):
        if hasattr(val_loss, "data"):
            val_loss = float(val_loss.data)

        #если это была первая эпоха
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint()

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint()

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    # восстанавливаем лучшие веса
                    self.model.load_state_dict(self.best_state_dict)

        return self.early_stop

    def _save_checkpoint(self): 
        if self.restore_best_weights and hasattr(self.model, "state_dict"):
            self.best_state_dict = copy.deepcopy(self.model.state_dict())


'''пример кода

from tinytensor.utils import EarlyStopping

model = MLP()
early_stopping = EarlyStopping(model, patience=7, min_delta=0.01)

for epoch in range(100):
    #обучение
    train_loss = train_one_epoch(model)

    #валидация
    val_loss = evaluate(model)

    #проверка остановки
    if early_stopping(val_loss):
        print(f"обучение прекратилось на эпохе {epoch+1}")
        break

'''