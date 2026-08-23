"""
Мини-GPT на tinytensor: char-level языковая модель, обучается на настоящем
tiny shakespeare корпусе (тот же датасет, что в оригинальном char-rnn/nanoGPT)
и в конце генерирует текст в стиле Шекспира.

Датасет лежит в examples/data/tinyshakespeare.txt. Если файла нет - скачайте:
curl -L -o examples/data/tinyshakespeare.txt \
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

Запуск: python examples/train_gpt.py
"""
import time

import numpy as np

from tinytensor.config import set_seed
from tinytensor.core.tensor import Tensor
from tinytensor.nn import (
    Module, Sequential, Embedding, LayerNorm, MultiHeadAttention,
    Linear, GELU, Dropout, CrossEntropyLoss,
)
from tinytensor.nn.utils import clip_grad_norm_
from tinytensor.optim import AdamW
from tinytensor.data import Dataset, DataLoader
from tinytensor.utils import train_bar

set_seed(0)

# ---- гиперпараметры (маленькие, чтоб обучалось на cpu за разумное время) ----
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
BLOCK_SIZE = 32
BATCH_SIZE = 32
DROPOUT = 0.1
LR = 3e-4
STEPS = 300
TRAIN_CHARS = 50_000  # подвыборка корпуса, для полного текста просто увеличьте


# ---- данные: посимвольная токенизация ----
with open("examples/data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()[:TRAIN_CHARS]

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return np.array([stoi[c] for c in s], dtype=np.int64)


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


data = encode(text)
print(f"корпус: {len(text)} символов, словарь: {vocab_size} уникальных символов")


class CharDataset(Dataset):
    # каждое окно - block_size символов на вход, те же символы сдвинутые на 1 - таргет
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y


dataset = CharDataset(data, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---- модель ----
class GPTBlock(Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout=dropout, causal=True)
        self.ln2 = LayerNorm(n_embd)
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.act = GELU()
        self.fc2 = Linear(4 * n_embd, n_embd)
        self.drop = Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        mlp_out = self.fc2(self.act(self.fc1(self.ln2(x))))
        x = x + self.drop(mlp_out)
        return x


class MiniGPT(Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(block_size, n_embd)
        # blocks через Sequential, а не голый python-список - иначе
        # Module.parameters()/to()/state_dict() их не увидят (см. docs/faq.md)
        self.blocks = Sequential(*[GPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size)

    def forward(self, idx):
        batch_size, seq_len = idx.data.shape
        pos_ids = Tensor(np.arange(seq_len, dtype=np.int64)[None, :].repeat(batch_size, axis=0))

        x = self.token_emb(idx) + self.pos_emb(pos_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


model = MiniGPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT)

n_params = sum(p.data.size for p in model.parameters())
print(f"модель: {n_params:,} параметров")

loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)


def generate(model, prompt, max_new_tokens=200, temperature=0.8):
    model.eval()
    ids = list(encode(prompt))

    for _ in range(max_new_tokens):
        context = ids[-model.block_size:]
        x = Tensor(np.array([context], dtype=np.int64))
        logits = model(x)

        last_logits = logits.data[0, -1, :] / temperature
        probs = np.exp(last_logits - last_logits.max())
        probs /= probs.sum()

        next_id = np.random.choice(len(probs), p=probs)
        ids.append(int(next_id))

    model.train()
    return decode(ids)


# ---- обучение ----
step = 0
t0 = time.time()
data_iter = iter(loader)

for step in train_bar(range(STEPS), prefix="обучение mini-gpt"):
    try:
        xb, yb = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        xb, yb = next(data_iter)

    logits = model(xb)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = Tensor(yb.data.reshape(-1))

    loss = loss_fn(logits_flat, targets_flat)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 50 == 0:
        print(f"\nstep {step:4d} | loss {float(loss.data):.4f}")

print(f"\nобучение заняло {time.time() - t0:.1f}s\n")

print("=" * 60)
print("генерация (модель маленькая и обучалась недолго, ждать шедевра не стоит):")
print("=" * 60)
print(generate(model, prompt="ROMEO:", max_new_tokens=300))
