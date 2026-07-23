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