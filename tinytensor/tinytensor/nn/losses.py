import numpy as np
from tinytensor.nn.modules import Module
# 1/n * sum((n, i = 1), yi - y`i)^2
class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        sq_diff = diff ** 2
        return sq_diff.sum() * (1.0/y_pred.data.size)



# мб в будущем добавлю cross entropy loss  