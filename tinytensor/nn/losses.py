import numpy as np
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
# 1/n * sum((n, i = 1), yi - y`i)^2
class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        sq_diff = diff ** 2
        return sq_diff.sum() * (1.0/y_pred.data.size)

# -log((e-xtar - M)/(sum(exj-M)))=-(xt-M)+log sum(exj-M)
class CrossEntropyLoss(Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps 

    def forward(self, logits, targets):
        x = logits.data
        batch_size = x.shape[0]

        max_logits = np.max(x, axis=1, keepdims=True)
        shifted_logits = x - max_logits
        
        #знаменатель софтмакс
        exp_shifted = np.exp(shifted_logits)
        sums_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        softmax = exp_shifted / sums_exp

        #подсчет лог
        #эпсилон чтоб на 0 не поделилось случайно
        log_softmax = shifted_logits - np.log(sums_exp + self.eps)

        if targets.data.ndim == 1 or targets.data.shape[1] == 1:
            target_indices = targets.data.astype(np.int64).ravel()
            correct_log_probs = log_softmax[np.arange(batch_size), target_indices]
            one_hot = np.zeros_like(x)
            one_hot[np.arange(batch_size), target_indices] = 1.0

        else:
            correct_log_probs = np.sum(log_softmax * targets.data, axis=1)
            one_hot = targets.data

        loss = -np.mean(correct_log_probs)

        out = Tensor(loss, requires_grad=logits.requires_grad, device=logits.device)

        if out.requires_grad:
            out._prev = {logits}

            def _backward():
                if logits.grad is None:
                    logits.grad = np.zeros_like(logits.data, dtype=np.float32)
                logits.grad += (softmax - one_hot) / batch_size * out.grad

            out._backward = _backward

        return out

