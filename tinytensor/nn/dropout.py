#функция дропаута это короче нужно чтоб вырубать нейроны во время обучения чтоб остальные нейроны обучились хорошо тоже

import numpy as np
from tinytensor.nn.modules import Module
from tinytensor.core.tensor import Tensor

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        # p это вероятност типа 0.5 это половина нейронов вырубается
        self.p = p

    def forward(self, x):
        # если не обучаемся то скип
        if not self.training or self.p == 0:
            return x
        mask_data = (np.random.rand(*x.data.shape) > self.p) / (1.0 - self.p)
        
        #маску в тензор превращаем
        mask = Tensor(mask_data, requires_grad=False)

        #тк маска состоит из 0 и 1 умножение как раз некоторые нейроны вырубит
        return x * mask
