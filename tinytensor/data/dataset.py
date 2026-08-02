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

def _download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


def _read_gz(path, header_size):
    with gzip.open(path, "rb") as f:
        data = f.read()             
    mass = np.frombuffer(data, dtype=np.uint8)   
    return mass[header_size:]             


#mnist 
def load_mnist(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)

    for key, filename in _MNIST_FILES.items():
        url = _MNIST_URL + filename
        path = os.path.join(data_dir, filename)
        _download(url, path)

    x_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_images"]), 16)
    y_train = _read_gz(os.path.join(data_dir, _MNIST_FILES["train_labels"]), 8)
    x_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_images"]), 16)
    y_test  = _read_gz(os.path.join(data_dir, _MNIST_FILES["test_labels"]), 8)
    
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    return (x_train, y_train), (x_test, y_test)
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



