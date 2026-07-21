import numpy as np

from tinytensor.core.autograd import backward as run_backward

def _unbroadcast(grad, target):
    while grad.ndim > len(target):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target):
        if dim == 1: 
            grad = grad.sum(axis=i, keepdims=True)
    return grad


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
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                #градиент для 2 параметра
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data, dtype=np.float32)

                    grad_other = out.grad * 1.0
                    other.grad += _unbroadcast(grad_other, other.data.shape)

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
                    grad_self = out.grad * other.data
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data, dtype=np.float32)
                    grad_other = out.grad * self.data
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward

        return out

    def __rmul__(self, other):
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

    #вычитание
    def __sub__(self, other):
        return self + (other * -1)
    
    def __rsub__(self, other):
        return (self * -1 ) + other
    
    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (power * (self.data ** (power - 1)))
            out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    #градиент будет распределен одинаково
                    self.grad += out.grad * np.ones_like(self.data)
            out._backward = _backward
        return out

    # когда это все прекратится?
    # что я делаю не так?
    # добавим функцию релу чтоб из активаторов можно было обращаться 
    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad
        )

        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (self.data > 0)

            out._backward = _backward

        return out

    def leaky_relu(self, alpha=0.01):
        out_data = np.where(self.data > 0, self.data, self.data * alpha)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    #производная=1 для x>0 и alpha для x<=0
                    dx = np.where(self.data > 0, 1.0, alpha)
                    self.grad += out.grad * dx
            out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.data, -50, 50)))
        out = Tensor(sig, requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (sig * (1.0 - sig))
            out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    #производная tanh 1-tanh^2
                    self.grad += out.grad * (1.0 - t ** 2)
            out._backward = _backward
        return out

    def gelu(self):
        #формула гелу 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        x = self.data
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
        out = Tensor(x * cdf, requires_grad=self.requires_grad)
        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data, dtype=np.float32)
                    pdf = np.exp(-0.5 * (x ** 2)) / np.sqrt(2.0 * np.pi)
                    d_gelu = cdf + x * pdf
                    self.grad += out.grad * d_gelu
            out._backward = _backward
        return out

    def backward(self):
        run_backward(self)
        

# ⣇⣿⠘⣿⣿⣿⡿⡿⣟⣟⢟⢟⢝⠵⡝⣿⡿⢂⣼⣿⣷⣌⠩⡫⡻⣝⠹⢿⣿⣷
# ⡆⣿⣆⠱⣝⡵⣝⢅⠙⣿⢕⢕⢕⢕⢝⣥⢒⠅⣿⣿⣿⡿⣳⣌⠪⡪⣡⢑⢝⣇
# ⡆⣿⣿⣦⠹⣳⣳⣕⢅⠈⢗⢕⢕⢕⢕⢕⢈⢆⠟⠋⠉⠁⠉⠉⠁⠈⠼⢐⢕⢽
# ⡗⢰⣶⣶⣦⣝⢝⢕⢕⠅⡆⢕⢕⢕⢕⢕⣴⠏⣠⡶⠛⡉⡉⡛⢶⣦⡀⠐⣕⢕
# ⡝⡄⢻⢟⣿⣿⣷⣕⣕⣅⣿⣔⣕⣵⣵⣿⣿⢠⣿⢠⣮⡈⣌⠨⠅⠹⣷⡀⢱⢕
# ⡝⡵⠟⠈⢀⣀⣀⡀⠉⢿⣿⣿⣿⣿⣿⣿⣿⣼⣿⢈⡋⠴⢿⡟⣡⡇⣿⡇⡀⢕
# ⡝⠁⣠⣾⠟⡉⡉⡉⠻⣦⣻⣿⣿⣿⣿⣿⣿⣿⣿⣧⠸⣿⣦⣥⣿⡇⡿⣰⢗⢄
# ⠁⢰⣿⡏⣴⣌⠈⣌⠡⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣬⣉⣉⣁⣄⢖⢕⢕⢕
# ⡀⢻⣿⡇⢙⠁⠴⢿⡟⣡⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣵⣵⣿
# ⡻⣄⣻⣿⣌⠘⢿⣷⣥⣿⠇⣿⣿⣿⣿⣿⣿⠛⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣷⢄⠻⣿⣟⠿⠦⠍⠉⣡⣾⣿⣿⣿⣿⣿⣿⢸⣿⣦⠙⣿⣿⣿⣿⣿⣿⣿⣿⠟
# ⡕⡑⣑⣈⣻⢗⢟⢞⢝⣻⣿⣿⣿⣿⣿⣿⣿⠸⣿⠿⠃⣿⣿⣿⣿⣿⣿⡿⠁⣠
# ⡝⡵⡈⢟⢕⢕⢕⢕⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣿⣿⣿⣿⣿⠿⠋⣀⣈⠙
# ⡝⡵⡕⡀⠑⠳⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⢉⡠⡲⡫⡪⡪⡣ 