import numpy as np

from tinytensor.core.tensor import Tensor, HAS_CUDA, cp
from tinytensor.nn.modules import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = str(device).lower()

        std = np.sqrt(2.0 / in_features)

        if HAS_CUDA and self.device == "cuda":
            w_data = (cp.random.randn(in_features, out_features) * std).astype(cp.float32)
            b_data = cp.zeros((1, out_features), dtype=cp.float32) if bias else None
        else:
            w_data = (np.random.randn(in_features, out_features) * std).astype(np.float32)
            b_data = np.zeros((1, out_features), dtype=np.float32) if bias else None

        self.weight = Tensor(w_data, requires_grad=True, device=self.device)
        self.bias = Tensor(b_data, requires_grad=True, device=self.device) if bias else None

    def forward(self, x):
        if getattr(self, "quantized", False):
            return self._forward_quant(x)
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    # ---------- structured pruning ----------

    def _prune_outputs(self, idx):
        # режем выходные нейроны (столбцы) текущего слоя
        new_w = np.delete(self.weight.data, idx, axis=1)
        self.weight = Tensor(new_w, requires_grad=True, device=self.device)
        if self.bias is not None:
            new_b = np.delete(self.bias.data, idx, axis=1)
            self.bias = Tensor(new_b, requires_grad=True, device=self.device)
        self.out_features = self.out_features - len(idx)

    def _prune_inputs(self, idx):
        new_w = np.delete(self.weight.data, idx, axis=0)
        self.weight = Tensor(new_w, requires_grad=True, device=self.device)
        self.in_features = self.in_features - len(idx)

    # ---------- int8 квантизация ----------

    @staticmethod
    def _quant_symmetric(arr, axis=None):
        # дробные числа -> целые int8 + scale
        # axis=None один scale на всё, axis=0 отдельный на каждый нейрон (точнее)
        amax = np.abs(arr).max(axis=axis, keepdims=(axis is not None))
        amax = np.where(amax == 0, 1.0, amax)          # чтоб не делить на 0
        scale = (amax / 127.0).astype(np.float32)
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        return q, scale

    def quant(self):
        # квантуем веса один раз, per-channel по нейронам
        w = np.asarray(self.weight.data, dtype=np.float32)
        self.w_q, self.w_scale = self._quant_symmetric(w, axis=0)
        self.b_data = None if self.bias is None else np.asarray(self.bias.data, dtype=np.float32)
        # выкидываем float веса чтоб освободить память
        self.weight = None
        self.bias = None
        self.quantized = True
        return self

    def _forward_quant(self, x):
        # вход квантуем на лету
        x_data = np.asarray(x.data, dtype=np.float32)
        x_q, sx = self._quant_symmetric(x_data, axis=None)

        # matmul в целых числах int8 @ int8 -> int32
        acc = x_q.astype(np.int32) @ self.w_q.astype(np.int32)

        # обратно во float
        y = acc.astype(np.float32) * sx * self.w_scale
        if self.b_data is not None:
            y = y + self.b_data

        return Tensor(y, requires_grad=False, device="cpu")

    def _quant_buffers(self):
        # что сохраняем в файл
        b = {"w_q": self.w_q, "w_scale": self.w_scale}
        if self.b_data is not None:
            b["b_data"] = self.b_data
        return b

    def _set_quant_buffers(self, sdict, prefix):
        # грузим int8 буферы обратно
        p = (prefix + ".") if prefix else ""
        if (p + "w_q") in sdict:
            self.w_q = sdict[p + "w_q"]
            self.w_scale = sdict[p + "w_scale"]
            self.b_data = sdict.get(p + "b_data", None)
            self.weight = None
            self.bias = None
            self.quantized = True
