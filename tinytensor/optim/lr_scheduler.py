#уменьшаем lr каждые n эпох чтоб модель обучилась более плавно
class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1 
        if self.last_epoch % self.step_size == 0:
            # уменьшаем условно в 10 раз
            self.optimizer.lr *= self.gamma
