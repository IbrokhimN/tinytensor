import numpy as np

from tinytensor.core.autograd import backward as run_backward

#чекаем ес можно на cuda матричные умножения сделать
try:
    from tinytensor import cuda_ops
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False



def _unbroadcast(grad, target):
    while grad.ndim > len(target):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# этот мир жесток
class Tensor:
    def __init__(self, data, requires_grad=False, device=None):
        # device=None -> берём глобальное устройство по умолчанию из config
        # (его можно менять через tinytensor.set_device('cuda'))
        if device is None:
            from tinytensor.config import config
            device = config.default_device
        self.device = device

        if isinstance(data, Tensor):
            self.data = data.data
            self.device = data.device
        elif HAS_CUDA and self.device == 'cuda':
            if isinstance(data, np.ndarray):
                self.data = cp.asarray(data, dtype=cp.float32)
            elif isinstance(data, cp.ndarray):
                self.data = data
            else:
                self.data = cp.array(data, dtype=cp.float32)
        else:
            if HAS_CUDA and isinstance(data, cp.ndarray):
                self.data = cp.asnumpy(data).astype(np.float32)
            elif isinstance(data, np.ndarray):
                self.data = data.astype(np.float32)
            else:
                self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def to(self, device):
        # перекидываем тензор на cuda или cpu
        device = str(device).lower()
        if device == self.device:
            return self
        
        if device == 'cuda':
            if not HAS_CUDA:
                raise RuntimeError("cuda или cupy недоступен")
            
            # с цпу на гпу
            new_data = cp.asarray(self.data)
            new_tensor = Tensor(new_data, requires_grad=self.requires_grad, device='cuda')
            if self.grad is not None:
                new_tensor.grad = cp.asarray(self.grad)
            return new_tensor
        
        elif device == 'cpu':
            if HAS_CUDA and isinstance(self.data, cp.ndarray):
                new_data = cp.asnumpy(self.data)
            else:
                new_data = self.data
                
            new_tensor = Tensor(new_data, requires_grad=self.requires_grad, device='cpu')
            if self.grad is not None and HAS_CUDA and isinstance(self.grad, cp.ndarray):
                new_tensor.grad = cp.asnumpy(self.grad)
            return new_tensor

        else:
            raise ValueError(f"нету устройства {device} напишите cpu или cuda")

            
    def __repr__(self):
        dev_str = f", device='{self.device}'" if self.device != 'cpu' else ""
        return f"Tensor({self.data}{dev_str}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    # сложение
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        if out.requires_grad:
            out._prev = {self, other}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)

                    grad_self = out.grad * 1.0
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                # градиент для 2 параметра
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = get_array_module(other.data).zeros_like(other.data, dtype=np.float32)

                    grad_other = out.grad * 1.0
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward

        return out

    def __radd__(self, other):
        # эт если пользователь решит написать не Tensor + 5 а 5 + Tensor
        return self + other

    # умножение
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        if out.requires_grad:
            out._prev = {self, other}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    grad_self = out.grad * other.data
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = get_array_module(other.data).zeros_like(other.data, dtype=np.float32)
                    grad_other = out.grad * self.data
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # матричное умножение
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        xp = get_array_module(self.data)

        # если девайс куда и бэкенд собрался то гоним через кублас а если нет то обычный нампай
        res_data = xp.matmul(self.data, other.data)

        out = Tensor(
            res_data,
            requires_grad=self.requires_grad or other.requires_grad,
            device=self.device,
        )

        if out.requires_grad:
            out._prev = {self, other}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    grad_self = xp.matmul(out.grad, other.data.swapaxes(-1, -2))
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = xp.zeros_like(other.data, dtype=np.float32)
                    grad_other = xp.matmul(self.data.swapaxes(-1, -2), out.grad)
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward

        return out

    # вычитание
    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return (self * -1) + other

    def __pow__(self, power):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (power * (self.data ** (power - 1)))

            out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * get_array_module(self.data).ones_like(self.data)

            out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad.reshape(self.data.shape)

            out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}
            inv_axes = tuple(np.argsort(axes))

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad.transpose(*inv_axes)

            out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            get_array_module(self.data).maximum(0, self.data),
            requires_grad=self.requires_grad,
            device=self.device,
        )

        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = get_array_module(self.data).zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (self.data > 0)

            out._backward = _backward

        return out

    def abs(self):
        xp = get_array_module(self.data)
        out = Tensor(xp.abs(self.data), requires_grad=self.requires_grad, device=self.device)

        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * xp.sign(self.data)
            out._backward = _backward
        return out

    def log(self):
        xp = get_array_module(self.data)
        out = Tensor(xp.log(self.data), requires_grad=self.requires_grad, device=self.device)

        if out.requires_grad:
            out._prev = {self}
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (1.0 / self.data)
            out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        xp = get_array_module(self.data)
        out_data = xp.where(self.data > 0, self.data, self.data * alpha)
        out = Tensor(out_data, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    dx = xp.where(self.data > 0, 1.0, alpha)
                    self.grad += out.grad * dx

            out._backward = _backward
        return out

    def sigmoid(self):
        xp = get_array_module(self.data)
        sig = 1.0 / (1.0 + xp.exp(-xp.clip(self.data, -50, 50)))
        out = Tensor(sig, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (sig * (1.0 - sig))

            out._backward = _backward
        return out

    def tanh(self):
        xp = get_array_module(self.data)
        t = xp.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    self.grad += out.grad * (1.0 - t ** 2)

            out._backward = _backward
        return out

    def gelu(self):
        xp = get_array_module(self.data)
        x = self.data
        cdf = 0.5 * (1.0 + xp.tanh(xp.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
        out = Tensor(x * cdf, requires_grad=self.requires_grad, device=self.device)
        if out.requires_grad:
            out._prev = {self}

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = xp.zeros_like(self.data, dtype=np.float32)
                    pdf = xp.exp(-0.5 * (x ** 2)) / xp.sqrt(2.0 * np.pi)
                    d_gelu = cdf + x * pdf
                    self.grad += out.grad * d_gelu

            out._backward = _backward
        return out

    def backward(self):
        run_backward(self)


def get_array_module(data):
    if hasattr(type(data), "__module__") and "cupy" in type(data).__module__:
        import cupy as cp
        return cp
    return np

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
# ⡝⡵⢟⢕⢕⢕⢕⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣿⣿⣿⣿⣿⠿⠋⣀⣈⠙
# ⡝⡵⡕⡀⠑⠳⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⢉⡠⡲⡫⡪⡪⡣



