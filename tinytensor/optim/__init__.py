from tinytensor.optim.optimizer import Optimizer, SGD, AdamW
from tinytensor.optim.lr_scheduler import StepLR, CosineAnnealingLR
__all__ = [
    "Optimizer", "SGD", "AdamW",
    "StepLR", "CosineAnnealingLR"
]

#эт для удобства
#чтоб можно было писать вот так:
"""
from tinytensor.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3)
"""
