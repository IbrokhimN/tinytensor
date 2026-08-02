from dataclasses import dataclass
import numpy as np

# проверяем доступна ли cuda/cupy
try:
    import cupy as cp
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False


@dataclass
class Config:
    # базовый сид будет 67
    seed: int = 67
    # устройство по умолчанию для новых тензоров и слоёв
    default_device: str = "cpu"


config = Config()


def set_seed(seed: int):
    # автоматически устанавливаем сид для воспроизведения и тд
    config.seed = seed
    np.random.seed(seed)


def set_device(device: str):
    device = str(device).lower()
    if device == "cuda" and not _HAS_CUDA:
        raise RuntimeError("cuda недоступна (cupy не установлен)")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"неизвестное устройство: {device}")
    config.default_device = device


def get_device():
    return config.default_device


def cuda_available():
    return _HAS_CUDA


# ну все при импорте будет устанавливаться
np.random.seed(config.seed)
