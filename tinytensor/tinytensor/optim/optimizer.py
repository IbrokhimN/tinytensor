# я допил свой чай, теперь хочу еще но мне лень вставать
import numpy as np
from tinytensor.core.tensor import get_array_module

class Optimizer:
    def __init__(self, parameters):
        # в лист если через генератор model параметры выдано
        self.parameters = list(parameters)

    def step(self):
        # у каждого свой
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


# SGD 
# w = w - lr*dw
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        #буферы под девайс каждого параметра
        self.velocities = []
        if momentum > 0:
            for p in self.parameters:
                xp = get_array_module(p.data)
                self.velocities.append(xp.zeros_like(p.data))
        else:
            self.velocities = None

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            grad = p.grad  
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]

            # шаг
            p.data -= self.lr * grad


# AdamW
class AdamW(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        #моменты под девайс каждого параметра
        self.m = []
        self.v = []
        for p in self.parameters:
            xp = get_array_module(p.data)
            self.m.append(xp.zeros_like(p.data))
            self.v.append(xp.zeros_like(p.data))

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            xp = get_array_module(p.data)
            grad = p.grad

            # обновка m и v моментов
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            # коррекция баяса
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            # адапт шаг и разделение затухания весов
            denom = xp.sqrt(v_hat) + self.eps
            
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

            p.data -= self.lr * (m_hat / denom)
