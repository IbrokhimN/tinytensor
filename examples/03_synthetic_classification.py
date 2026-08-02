import numpy as np

from tinytensor.core.tensor import Tensor
from tinytensor.nn.linear import Linear
from tinytensor.nn.dropout import Dropout
from tinytensor.nn import Sequential
from tinytensor.nn.losses import CrossEntropyLoss
from tinytensor.optim.optimizer import AdamW

# 1. Генерация синтетического датасета (3 кластера точек)
def generate_synthetic_data(num_samples=900, num_features=10, num_classes=3):
    np.random.seed(42)
    samples_per_class = num_samples // num_classes
    
    X_list, y_list = [], []
    for c in range(num_classes):
        # Генерируем сдвинутые случайные данные для каждого класса
        center = np.random.randn(num_features) * 2.0
        x_class = np.random.randn(samples_per_class, num_features) + center
        y_class = np.full(samples_per_class, c)
        
        X_list.append(x_class)
        y_list.append(y_class)
        
    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.int64)
    
    # Перемешиваем
    indices = np.random.permutation(num_samples)
    return X[indices], y[indices]


def main():
    device = "cpu"  # Можно изменить на "cuda", если C++ CUDA модуль собран
    
    print("--- 1. Подготовка данных ---")
    X_raw, y_raw = generate_synthetic_data(num_samples=900, num_features=10, num_classes=3)
    
    # Делим на train / val (80% / 20%)
    split = int(0.8 * len(X_raw))
    X_train, y_train = X_raw[:split], y_raw[:split]
    X_val, y_val = X_raw[split:], y_raw[split:]

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    print("\n--- 2. Создание модели через Sequential ---")
    model = Sequential(
        Linear(in_features=10, out_features=32, device=device),
        Dropout(p=0.1),
        Linear(in_features=32, out_features=16, device=device),
        Linear(in_features=16, out_features=3, device=device)
    )

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    print("\n--- 3. Цикл обучения ---")
    epochs = 40
    batch_size = 32
    num_batches = len(X_train) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()  # Включаем Dropout для обучения
        total_loss = 0.0
        
        # Перемешивание батчей
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            xb = Tensor(X_train_shuffled[start:end], device=device)
            yb = Tensor(y_train_shuffled[start:end], device=device)

            # Forward
            logits = model(xb)
            loss = criterion(logits, yb)

            # Backward & Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item() if hasattr(loss.data, "item") else float(loss.data)

        avg_train_loss = total_loss / num_batches

        # Оценка на валидации
        model.eval()  # Отключаем Dropout для инференса
        val_x = Tensor(X_val, device=device)
        val_y = Tensor(y_val, device=device)
        
        val_logits = model(val_x)
        val_loss = criterion(val_logits, val_y)
        
        # Точность (Accuracy)
        preds = np.argmax(val_logits.data, axis=1)
        acc = np.mean(preds == y_val) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.data:.4f} | Val Acc: {acc:.2f}%")

    print("\n✅ Обучение успешно завершено!")


if __name__ == "__main__":
    main()

