import os
import gzip
import urllib.request
import numpy as np
# ЮРЛы
_MNIST_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
def _download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
def _read_gz(path, header_size):
    with gzip.open(path, "rb") as f:
        data = f.read()             
    mass = np.frombuffer(data, dtype=np.uint8)   
    return mass[header_size:]             
#mnist 
def load_mnist(data_dir="./data_mnist", normalize=True):
    os.makedirs(data_dir, exist_ok=True)
    for key, filename in _MNIST_FILES.items():
        url = _MNIST_URL + filename
        path = os.path.join(data_dir, filename)
        _download(url, path)
    x_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_images"]), 16)
    y_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_labels"]), 8)
    x_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_images"]), 16)
    y_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_labels"]), 8)

    if normalize:
        x_train = (x_train.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
        x_test  = (x_test.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
    else:
        x_train = x_train.reshape(-1, 28, 28)
        x_test  = x_test.reshape(-1, 28, 28)
    return (x_train, y_train), (x_test, y_test)
#fashion
def load_fashion(data_dir="./data_fashion", normalize=True):
    os.makedirs(data_dir, exist_ok=True)
    for key, filename in _MNIST_FILES.items():
        url = _FASHION_URL + filename
        path = os.path.join(data_dir, filename)
        _download(url, path)
    x_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_images"]), 16)
    y_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_labels"]), 8)
    x_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_images"]), 16)
    y_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_labels"]), 8)

    if normalize:
        x_train = (x_train.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
        x_test  = (x_test.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
    else:
        x_train = x_train.reshape(-1, 28, 28)
        x_test  = x_test.reshape(-1, 28, 28)
    return (x_train, y_train), (x_test, y_test)
import csv as _csv
#загрузчик табличных данных из csv
def load_csv(path, target=-1, has_header=True, delimiter=",",
             normalize=False, test_split=None, shuffle=True, seed=None):
    with open(path, "r", newline="") as f:
        reader = _csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    header = None
    if has_header:
        header = rows[0]
        rows = rows[1:]

    # target можно задать именем колонки
    if isinstance(target, str):
        if header is None:
            raise ValueError("target задан именем, но has_header=False")
        target_idx = header.index(target)
    else:
        n_cols = len(rows[0])
        target_idx = target % n_cols

    x_raw, y_raw = [], []
    for r in rows:
        y_raw.append(r[target_idx])
        feats = [v for i, v in enumerate(r) if i != target_idx]
        x_raw.append(feats)

    x = np.array(x_raw, dtype=np.float32)

    # метки числами, если не выходит - кодируем строки в 0,1,2...
    try:
        y = np.array(y_raw, dtype=np.int64)
    except ValueError:
        classes = sorted(set(y_raw))
        mapping = {name: i for i, name in enumerate(classes)}
        y = np.array([mapping[v] for v in y_raw], dtype=np.int64)
        print(f"строковые метки закодированы: {mapping}")

    # z-score по каждому столбцу
    if normalize:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1.0          # чтоб не делить на ноль
        x = (x - mean) / std

    if test_split is None:
        return x, y

    # делим на train/test
    n = len(x)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = int(n * test_split)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])

#влом коменты ставить тут и так все понятно
class Dataset:
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
class TensorDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
