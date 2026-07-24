from tinytensor.core.tensor import Tensor
import pickle

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
        
    def state_dict(self):
        #вытащим все веса параметров
        sdict = {}
        for name, param in self._get_named_params().items():
            sdict[name] = param.data
        return sdict
        
    def load_state_dict(self, sdict):
        #запихиваем обратно
        for name, param in self._get_named_params().items():
            if name in sdict:
                param.data = sdict[name] 

    def _get_named_params(self, prefix=""):
        named = {}
        for attr_name, attr in self.__dict__.items():
            key = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(attr, Tensor):
                named[key] = attr
            elif isinstance(attr, Module):
                named.update(attr._get_named_params(prefix=key))
        return named 
    
    #сохранение модели
    def save(self,filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.state_dict(), f)

    # загрузка весов обратно
    def load(self, filepath):
        with open(filepath, "rb") as f:
            sd = pickle.load(f)
            self.load_state_dict(sd)
