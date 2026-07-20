import numpy as np

from tinytensor.core.autograd import backward as run_backward

# этот мир жесток
class Tensor:
    def __init__(self, data, requires_grad=False):
            # кароч такая штука типа, если передалось уже нампай массив
            # то мы сразу сохраняем его так
            # а если не нампай массив а просто массив то тогда переводим его в нампай массив
            if isinstance(data, np.ndarray):
                #ставим базовый тип данных
                self.data = data.astype(np.float32) 
            else:
                #тут тоже
                self.data = np.array(data, dtype=np.float32)
                  
            self.grad = None
            # по умолчанию это будет пустой функцией
            self._backward = lambda: None
            # предки тензора в графе
            self._prev = set()
            self.requires_grad = requires_grad
            

    def __repr__(self):
            return f"tensor({self.data}, requires_grad={self.requires_grad})"
      
    @property
    def shape(self):
        return self.data.shape
    

    # сложение
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            out._prev = {self, other}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)

                    grad_self = out.grad * 1.0
                    while grad_self.ndim > self.data.ndim:
                        grad_self = grad_self.sum(axis=0)
                    for i, dim in enumerate(self.data.shape):
                        if dim == 1:
                            grad_self = grad_self.sum(axis=i, keepdims=True)
                    self.grad += grad_self

                #градиент для 2 параметра
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data, dtype=np.float32)

                    grad_other = out.grad * 1.0
                    while grad_other.ndim > other.data.ndim:
                        grad_other = grad_other.sum(axis=0)
                    for i, dim in enumerate(other.data.shape):
                        if dim == 1:
                            grad_other = grad_other.sum(axis=i, keepdims=True)
                    other.grad += grad_other
            out._backward = _backward

        return out 
    def __radd__(self, other):
            # эт если пользователь решит написать не Tensor + 5 а 5 + Tensor
            return self + other

    # умножение
    # тут идет просто тот же ход мыслей что и в +
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            # тут мы типа высчитываем градиент для будущей колибровки параметров обучения
            out._prev = {self, other}
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * other.data

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data, dtype=np.float32)
                    other.grad += out.grad * self.data

            out._backward = _backward
    
    def __rmul__(self,other):
        return self * other
        #сново переводим это на обычный mul
            
    # матричное умножение
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if out.requires_grad:
            out._prev = {self, other}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    #это формула производной матричного умножения
                    self.grad += np.matmul(
                        out.grad, other.data.T
                    )  # .T эт кароче переворот матрицы транспонирование вродь называется

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data, dtype=np.float32)
                    # это формула производной матричного умножения dL/dB = AT @ dL/dOut
                    other.grad += np.matmul(self.data.T, out.grad)

            out._backward = _backward

        return out
    # обратно переводить его нельзя из за правила линейной алгребы того что A*B != B*A, и тупо размерности не подойдут

    def backward(self):
          run_backward(self)
# tens = Tensor([1,2,3,4,5])
# tens2 = Tensor([6,7,8,9], requires_grad=True)
# print(tens)
# print(tens2)
# print(tens2.shape)
# tensor([1 2 3 4 5], requires_grad=False)
# tensor([6 7 8 9], requires_grad=True)
# (4, )

# все работает