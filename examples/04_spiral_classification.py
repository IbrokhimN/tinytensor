import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU
from tinytensor.nn import Sequential
from tinytensor.nn.losses import CrossEntropyLoss
from tinytensor.optim.optimizer import AdamW


# Генерация 3 спиралей на плоскости
def generate_spiral_data(samples_per_class=300, num_classes=3):
    np.random.seed(42)
    N, K = samples_per_class, num_classes
    X = np.zeros((N * K, 2), dtype=np.float32)
    y = np.zeros(N * K, dtype=np.int64)

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    indices = np.random.permutation(N * K)
    return X[indices], y[indices]


def main():
    device = "cuda"
    print(device)
    X_raw, y_raw = generate_spiral_data(samples_per_class=300, num_classes=3)
    
    split = int(0.8 * len(X_raw))
    X_train, y_train = X_raw[:split], y_raw[:split]
    X_val, y_val = X_raw[split:], y_raw[split:]

    # ТУТ ДОБАВЛЕНЫ ReLU — теперь это реально нелинейная нейросеть
    model = Sequential(
        Linear(in_features=2, out_features=64, device=device),
        ReLU(),
        Linear(in_features=64, out_features=64, device=device),
        ReLU(),
        Linear(in_features=64, out_features=3, device=device)
    )

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

    epochs = 150
    batch_size = 64
    num_batches = len(X_train) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        perm = np.random.permutation(len(X_train))
        X_tr_s, y_tr_s = X_train[perm], y_train[perm]

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            xb = Tensor(X_tr_s[start:end], device=device)
            yb = Tensor(y_tr_s[start:end], device=device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item() if hasattr(loss.data, "item") else float(loss.data)

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            val_x = Tensor(X_val, device=device)
            val_logits = model(val_x)
            preds = np.argmax(val_logits.data, axis=1)
            acc = np.mean(preds == y_val) * 100.0
            print(f"Epoch [{epoch:03d}/{epochs}] | Loss: {total_loss/num_batches:.4f} | Val Acc: {acc:.2f}%")


if __name__ == "__main__":
    main()
