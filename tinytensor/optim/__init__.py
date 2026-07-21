from tinytensor.optim.optimizer import Optimizer, SGD, AdamW

__all__ = ["Optimizer", "SGD", "AdamW"]

#эт для удобства
#чтоб можно было писать вот так:
"""
from tinytensor.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3)
"""