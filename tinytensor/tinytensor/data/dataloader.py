import numpy as np
from tinytensor.core.tensor import Tensor

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        #колво батчей за 1 эпоху
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        
        # перемешкиваем если шафл есть
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        self.current_idx = 0
        return self

    def __next__(self):
        # выдает след батч
        # останавливаемся если дошли до конца датасета
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        #cбор
        batch_x, batch_y = [], []
        for i in batch_indices:
            x, y = self.dataset[i]
            batch_x.append(x.data if isinstance(x, Tensor) else x)
            batch_y.append(y.data if isinstance(y, Tensor) else y)

        # батч в нампай и потом в тензо
        return Tensor(np.array(batch_x)), Tensor(np.array(batch_y))