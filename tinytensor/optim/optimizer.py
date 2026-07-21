# я допил свой чай, теперь хочу еще но мне лень вставать
import numpy as np

class Optimizer:
    def __init__ (self, parameters):
        # в лист если через генератор model параметры выдано
        self.parameters = list(parameters)

    def step(self):
        #у каждого свой
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

# SGD 
# w = w - lr*dw
class SGD(Optimizer):
    def __init__(self, parameters, lr = 0.01, momentum = 0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        # буфера
        self.velocities = [np.zeros_like(p.data) for p in self.parameters] if momentum > 0 else None

    def step(self):
        for i,p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad

            if self.momentum>0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            # шаг
            p.data -= self.lr * grad



#Adamw
# слишком много формул мне впадлу писать
class AdamW(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad
            # обновка m и v мометы
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            # корекция баяса
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            # адапт шаг и разделение затухания весов
            denom = np.sqrt(v_hat) + self.eps
            p.data -= self.lr * (m_hat / denom + self.weight_decay * p.data)