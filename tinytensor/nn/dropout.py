#функция дропаута это короче нужно чтоб вырубать нейроны во время обучения чтоб остальные нейроны обучились хорошо тоже

from tinytensor.nn.modules import Module
from tinytensor.core.tensor import Tensor, get_array_module

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        # p это вероятност типа 0.5 это половина нейронов вырубается
        self.p = p

    def forward(self, x):
        # если не обучаемся то скип
        if not self.training or self.p == 0:
            return x
            
        xp = get_array_module(x.data)
        mask_data = (xp.random.rand(*x.data.shape) > self.p) / (1.0 - self.p)
        
        #маску в тензор превращаем (указываем device чтобы не упало на cuda)
        mask = Tensor(mask_data, requires_grad=False, device=x.device)

        #тк маска состоит из 0 и 1 умножение как раз некоторые нейроны вырубит
        return x * mask
