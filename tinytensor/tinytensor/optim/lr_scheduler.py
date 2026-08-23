import math

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


#плавно снижаем lr по косинусу от стартового до eta_min за T_max эпох
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr   # запоминаем стартовый lr (это lr_max)
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        # lr = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(pi * epoch / T_max))
        cos = math.cos(math.pi * self.last_epoch / self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + cos)
