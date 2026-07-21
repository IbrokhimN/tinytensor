from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    # базовый сид будет 67
    seed: int = 67


config = Config()


def set_seed(seed: int):
    # автоматически устанавливаем сид для воспроизведения и тд
    config.seed = seed
    np.random.seed(seed)

# ну все при импорте будет устанавливаться
np.random.seed(config.seed)