from tinytensor.core.tensor import Tensor

#кароч модуль от которого будут наследоваться все слои
class Module:
    def __init__(self):
        self.training = True
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    #щас это просто заглушка для прямого вызова, а в реале у всех должны быть свои реализации forward, так что это роль не играет
    def forward(self, *args, **kwargs):
        raise NotImplementedError(...)

    # сохраняет обучаемые веса, и если в модуле есть еще какие то модули то он их тоже открывает 
    def parameters(self):
        params = []
        for atr in self.__dict__.values():
            if isinstance(atr, Tensor) and atr.requires_grad == True:
                params.append(atr)

            elif isinstance(atr, Module):
                params.extend(atr.parameters())
        return params

    #обнуляет градиенты
    def zero_grad(self):
        for atr in self.parameters():
            atr.grad = None
        
        
             