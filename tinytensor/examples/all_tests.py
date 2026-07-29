import numpy as np
from tinytensor.core.tensor import Tensor
from tinytensor.core.autograd import backward
from tinytensor.nn.modules import Module, Sequential
from tinytensor.nn.linear import Linear
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.batchnorm import BatchNorm2d
from tinytensor.nn.rnn import RNN, RNNCell
from tinytensor.nn.activations import ReLU, LeReLU, Sigmoid, Tanh, GELU, Softmax
from tinytensor.nn.dropout import Dropout
from tinytensor.nn.losses import MSELoss, CrossEntropyLoss
from tinytensor.nn.pooling import MaxPool2d, AvgPool2d
from tinytensor.optim.optimizer import SGD, AdamW
from tinytensor.data.dataloader import DataLoader
from tinytensor.data.dataset import Dataset
from tinytensor.utils.bar import progress_bar
from tinytensor.utils.early_stopping import EarlyStopping
from tinytensor.utils.summary import summary

print("=" * 80)
print("TINYTENSOR COMPREHENSIVE EXAMPLE - ДЕМОНСТРАЦИЯ ВСЕХ КОМПОНЕНТОВ")
print("=" * 80)

print("\n" + "█" * 80)
print("1. TENSOR ОПЕРАЦИИ")
print("█" * 80)

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f"\nТензор a: {a}")
print(f"Тензор b: {b}")

c = a + b
print(f"a + b = {c.data}")

d = a * b
print(f"a * b = {d.data}")

e = a - b
print(f"a - b = {e.data}")

f = a ** 2
print(f"a ** 2 = {f.data}")

g = (a + b).sum()
print(f"(a + b).sum() = {g.data}")

g.backward()
print(f"После backward():")
print(f"  a.grad = {a.grad}")
print(f"  b.grad = {b.grad}")

print("\n✓ Tensor операции работают!")

print("\n" + "█" * 80)
print("2. МАТРИЧНОЕ УМНОЖЕНИЕ И AUTOGRAD")
print("█" * 80)

x = Tensor(np.random.randn(2, 3), requires_grad=True)
w = Tensor(np.random.randn(3, 4), requires_grad=True)

y = x @ w
print(f"\nx shape: {x.shape}")
print(f"w shape: {w.shape}")
print(f"y = x @ w shape: {y.shape}")

loss = y.sum()
loss.backward()
print(f"\nПосле backward():")
print(f"  x.grad shape: {x.grad.shape}")
print(f"  w.grad shape: {w.grad.shape}")

print("\n✓ Matmul и autograd работают!")

print("\n" + "█" * 80)
print("3. АКТИВАЦИОННЫЕ ФУНКЦИИ")
print("█" * 80)

x = Tensor(np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]), requires_grad=True)
print(f"Input: {x.data}")

print(f"\nReLU: {ReLU().forward(x).data}")
print(f"LeReLU: {LeReLU(alpha=0.1).forward(x).data}")
print(f"Sigmoid: {Sigmoid().forward(x).data}")
print(f"Tanh: {Tanh().forward(x).data}")
print(f"GELU: {GELU().forward(x).data}")

x_for_softmax = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
softmax_out = Softmax(dim=1).forward(x_for_softmax)
print(f"Softmax: {softmax_out.data}")
print(f"Sum of softmax (должно быть 1.0): {softmax_out.data.sum()}")

print("\n✓ Активации работают!")

print("\n" + "█" * 80)
print("4. LINEAR СЛОЙ")
print("█" * 80)

linear = Linear(in_features=10, out_features=5, bias=True)
x_linear = Tensor(np.random.randn(2, 10), requires_grad=True)
out_linear = linear.forward(x_linear)

print(f"Input shape: {x_linear.shape}")
print(f"Linear layer output shape: {out_linear.shape}")
print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape if linear.bias is not None else 'None'}")

loss = out_linear.sum()
loss.backward()
print(f"After backward:")
print(f"  linear.weight.grad shape: {linear.weight.grad.shape}")
print(f"  linear.bias.grad shape: {linear.bias.grad.shape if linear.bias.grad is not None else 'None'}")

print("\n✓ Linear слой работает!")

print("\n" + "█" * 80)
print("5. CONV2D СЛОЙ")
print("█" * 80)

conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
x_conv = Tensor(np.random.randn(2, 3, 32, 32), requires_grad=True)
out_conv = conv.forward(x_conv)

print(f"Input shape (N, C, H, W): {x_conv.shape}")
print(f"Conv2d output shape: {out_conv.shape}")
print(f"Kernel shape: {conv.weight.shape}")
print(f"Bias shape: {conv.bias.shape}")

loss = out_conv.sum()
loss.backward()
print(f"After backward:")
print(f"  conv.weight.grad shape: {conv.weight.grad.shape}")

print("\n✓ Conv2d слой работает!")

print("\n" + "█" * 80)
print("6. BATCH NORMALIZATION")
print("█" * 80)

batchnorm = BatchNorm2d(num_features=16)
x_bn = Tensor(np.random.randn(4, 16, 8, 8), requires_grad=True)

batchnorm.train(True)
out_bn_train = batchnorm.forward(x_bn)
print(f"BatchNorm2d output shape (train): {out_bn_train.shape}")

batchnorm.eval()
out_bn_eval = batchnorm.forward(x_bn)
print(f"BatchNorm2d output shape (eval): {out_bn_eval.shape}")

print("\n✓ BatchNorm2d работает!")

print("\n" + "█" * 80)
print("7. DROPOUT")
print("█" * 80)

dropout = Dropout(p=0.5)
x_drop = Tensor(np.ones((2, 10)), requires_grad=True)

dropout.train(True)
out_drop_train = dropout.forward(x_drop)
print(f"Dropout при обучении:")
print(f"  Входной shape: {x_drop.shape}")
print(f"  Выходной shape: {out_drop_train.shape}")
print(f"  Примерная доля активных нейронов: {(out_drop_train.data > 0).sum() / out_drop_train.data.size:.2%}")

dropout.eval()
out_drop_eval = dropout.forward(x_drop)
print(f"Dropout при eval (без dropout):")
print(f"  Выход совпадает с входом: {np.allclose(out_drop_eval.data, x_drop.data)}")

print("\n✓ Dropout работает!")

print("\n" + "█" * 80)
print("8. RNN СЛОЙ")
print("█" * 80)

rnn = RNN(input_size=5, hidden_size=8)
x_rnn = Tensor(np.random.randn(2, 4, 5), requires_grad=True)
out_rnn = rnn.forward(x_rnn)

print(f"Input shape (batch, seq_len, input_size): {x_rnn.shape}")
print(f"RNN output shape: {out_rnn.shape}")

print("\n✓ RNN слой работает!")

print("\n" + "█" * 80)
print("9. SEQUENTIAL МОДЕЛЬ")
print("█" * 80)

model = Sequential(
    Linear(10, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 2)
)

x_seq = Tensor(np.random.randn(4, 10), requires_grad=True)
out_seq = model.forward(x_seq)

print(f"Sequential model:")
print(f"  Input shape: {x_seq.shape}")
print(f"  Output shape: {out_seq.shape}")
print(f"  Number of layers: {len(model)}")

total_params = sum(int(np.prod(p.data.shape)) for p in model.parameters())
print(f"  Total parameters: {total_params}")

print("\n✓ Sequential модель работает!")

print("\n" + "█" * 80)
print("10. ФУНКЦИИ ПОТЕРЬ")
print("█" * 80)

y_pred = Tensor(np.random.randn(4, 3), requires_grad=True)
y_true = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))

mse_loss = MSELoss()
mse_out = mse_loss.forward(y_pred, y_true)
print(f"MSE Loss: {mse_out.data}")

y_logits = Tensor(np.random.randn(4, 3), requires_grad=True)
y_targets = Tensor(np.array([0, 1, 2, 0]))

ce_loss = CrossEntropyLoss()
ce_out = ce_loss.forward(y_logits, y_targets)
print(f"CrossEntropy Loss: {ce_out.data}")

print("\n✓ Функции потерь работают!")

print("\n" + "█" * 80)
print("11. ОПТИМИЗАТОРЫ")
print("█" * 80)

model_opt = Sequential(
    Linear(5, 10),
    ReLU(),
    Linear(10, 2)
)

sgd = SGD(model_opt.parameters(), lr=0.01, momentum=0.9)
adamw = AdamW(model_opt.parameters(), lr=0.001, weight_decay=0.01)

x_opt = Tensor(np.random.randn(4, 5), requires_grad=True)
y_opt = Tensor(np.random.randn(4, 2), requires_grad=True)

out_opt = model_opt.forward(x_opt)
loss_opt = ((out_opt - y_opt) ** 2).sum()
loss_opt.backward()

print(f"Перед step optimizer'а:")
print(f"  model_opt.parameters()[0] (первый элемент): {model_opt.parameters()[0].data.flat[0]}")

sgd.step()
print(f"После SGD step: {model_opt.parameters()[0].data.flat[0]}")

model_opt.zero_grad()
print(f"После zero_grad: градиент первого параметра: {model_opt.parameters()[0].grad}")

print("\n✓ Оптимизаторы работают!")

print("\n" + "█" * 80)
print("12. DATALOADER")
print("█" * 80)

class SimpleDataset(Dataset):
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.x = np.random.randn(n_samples, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return Tensor(self.x[idx]), Tensor(self.y[idx])

dataset = SimpleDataset(n_samples=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Dataloader batches per epoch: {len(dataloader)}")

for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    if batch_idx == 0:
        print(f"First batch:")
        print(f"  X shape: {x_batch.shape}")
        print(f"  Y shape: {y_batch.shape}")
    if batch_idx == 1:
        break

print("\n✓ DataLoader работает!")

print("\n" + "█" * 80)
print("13. PROGRESS BAR И УТИЛИТЫ")
print("█" * 80)

print("Progress bar example:")
for i in progress_bar(range(20), prefix="Processing"):
    pass

print("\n✓ Progress bar работает!")

print("\n" + "█" * 80)
print("14. EARLY STOPPING")
print("█" * 80)

model_es = Sequential(Linear(5, 3))
early_stopping = EarlyStopping(model_es, patience=3, min_delta=0.001)

print("Simulating validation losses:")
val_losses = [0.5, 0.45, 0.44, 0.44, 0.445, 0.45]
for epoch, val_loss in enumerate(val_losses):
    should_stop = early_stopping(val_loss)
    print(f"  Epoch {epoch + 1}: loss = {val_loss}, stop = {should_stop}")
    if should_stop:
        print("  -> EarlyStopping triggered!")
        break

print("\n✓ EarlyStopping работает!")

print("\n" + "█" * 80)
print("15. MODEL SUMMARY")
print("█" * 80)

model_summary = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
)

summary(model_summary, input_shape=(1, 784))

print("\n" + "█" * 80)
print("16. ПОЛНЫЙ TRAINING LOOP ПРИМЕР")
print("█" * 80)

class SimpleNN(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 16)
        self.relu1 = ReLU()
        self.fc2 = Linear(16, 8)
        self.relu2 = ReLU()
        self.fc3 = Linear(8, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def parameters(self):
        params = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            params.extend(layer.parameters())
        return params

class TrainDataset(Dataset):
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        self.x = np.random.randn(n_samples, 10).astype(np.float32)
        self.y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return Tensor(self.x[idx]), Tensor(self.y[idx])

train_dataset = TrainDataset(200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_train = SimpleNN()
optimizer = AdamW(model_train.parameters(), lr=0.001)
loss_fn = MSELoss()

print("\nTraining for 3 epochs:")
for epoch in range(3):
    epoch_loss = 0
    batch_count = 0
    
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        y_pred = model_train.forward(x_batch)
        y_batch_reshaped = Tensor(y_batch.data.reshape(-1, 1))
        loss = loss_fn.forward(y_pred, y_batch_reshaped)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += float(loss.data)
        batch_count += 1
    
    avg_loss = epoch_loss / batch_count
    print(f"  Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

print("\n✓ Полный training loop работает!")

print("\n" + "=" * 80)
print("ВСЕ КОМПОНЕНТЫ TINYTENSOR УСПЕШНО ПРОТЕСТИРОВАНЫ!")
print("=" * 80)
