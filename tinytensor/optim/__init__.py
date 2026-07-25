from tinytensor.optim.optimizer import Optimizer, SGD, AdamW
from tinytensor.optim.lr_scheduler import StepLR
__all__ = [
    "Optimizer", "SGD", "AdamW",
    "StepLR"
]

#эт для удобства
#чтоб можно было писать вот так:
"""
from tinytensor.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3)
"""
