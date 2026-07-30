from tinytensor.core.tensor import Tensor, get_array_module
from tinytensor.nn.modules import Module

# 1/n * sum((n, i = 1), yi - y`i)^2
class MSELoss(Module):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        sq_diff = diff ** 2
        return sq_diff.sum() * (1.0 / y_pred.data.size)

# -log((e-xtar - M)/(sum(exj-M)))=-(xt-M)+log sum(exj-M)
class CrossEntropyLoss(Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps 

    def forward(self, logits, targets):
        x = logits.data
        xp = get_array_module(x) # Магия: достаем numpy или cupy в зависимости от девайса
        batch_size = x.shape[0]

        max_logits = xp.max(x, axis=1, keepdims=True)
        shifted_logits = x - max_logits
        
        # знаменатель софтмакс
        exp_shifted = xp.exp(shifted_logits)
        sums_exp = xp.sum(exp_shifted, axis=1, keepdims=True)
        softmax = exp_shifted / sums_exp

        # подсчет лог
        # эпсилон чтоб на 0 не поделилось случайно
        log_softmax = shifted_logits - xp.log(sums_exp + self.eps)

        if targets.data.ndim == 1 or targets.data.shape[1] == 1:
            target_indices = targets.data.astype(int).ravel()
            #индексация работает одинаково в numpy и cupy
            correct_log_probs = log_softmax[xp.arange(batch_size), target_indices]
            one_hot = xp.zeros_like(x)
            one_hot[xp.arange(batch_size), target_indices] = 1.0
        else:
            correct_log_probs = xp.sum(log_softmax * targets.data, axis=1)
            one_hot = targets.data

        loss = -xp.mean(correct_log_probs)

        out = Tensor(loss, requires_grad=logits.requires_grad, device=logits.device)

        if out.requires_grad:
            out._prev = {logits}

            def _backward():
                if logits.grad is None:
                    logits.grad = xp.zeros_like(logits.data, dtype=xp.float32)
                logits.grad += (softmax - one_hot) / batch_size * out.grad

            out._backward = _backward

        return out
