import numpy as np

from tinytensor.data import TensorDataset, DataLoader


def make_dataset(n=10):
    x = np.arange(n).reshape(n, 1).astype(np.float32)
    y = (np.arange(n) * 2).astype(np.float32)
    return TensorDataset(x, y)


def test_tensor_dataset_len_and_getitem():
    ds = make_dataset(5)
    assert len(ds) == 5
    x, y = ds[2]
    assert x[0] == 2.0
    assert y == 4.0


def test_dataloader_batches_cover_full_dataset():
    ds = make_dataset(10)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    seen = 0
    for xb, yb in loader:
        seen += xb.shape[0]
    assert seen == 10


def test_dataloader_len_is_ceil_division():
    ds = make_dataset(10)
    loader = DataLoader(ds, batch_size=3)
    assert len(loader) == 4  # ceil(10/3)


def test_dataloader_last_batch_smaller():
    ds = make_dataset(10)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batches = list(loader)
    assert batches[-1][0].shape[0] == 1  # 10 = 3+3+3+1


def test_dataloader_shuffle_changes_order_with_fixed_seed():
    ds = make_dataset(20)
    np.random.seed(0)
    loader = DataLoader(ds, batch_size=20, shuffle=True)
    xb, _ = next(iter(loader))
    order = xb.data.flatten()
    assert not np.array_equal(order, np.arange(20))  # почти наверняка перемешано
