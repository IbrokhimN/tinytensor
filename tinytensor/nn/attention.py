import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import Softmax
from tinytensor.nn.dropout import Dropout

# механизм внимания Q,K,V, скор = Q@K^T / sqrt(d_k), softmax, @V
# несколько голов считаются параллельно одной матрице без питон цикла по головам
class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim должен делиться на num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.attn_dropout = Dropout(dropout)
        self.softmax = Softmax(dim=-1)

    def _split_heads(self, x, batch_size, seq_len):
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x, batch_size, seq_len):
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.data.shape

        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_dim))

        if self.causal:
            causal_bias = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
            scores = scores + Tensor(causal_bias, requires_grad=False, device=x.device)

        if mask is not None:
            scores = scores + mask

        attn = self.softmax(scores)
        attn = self.attn_dropout(attn)

        out = attn @ v                                  # (batch, heads, seq, head_dim)
        out = self._merge_heads(out, batch_size, seq_len)

        return self.out_proj(out)
