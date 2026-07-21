from tinytensor.nn.modules import Module

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class LeReLU(Module):
    # у нас есть тут альфа
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return x.leaky_relu(alpha=self.alpha)
    
class Sigmod(Module):
    def forward(self, x):
        return x.sigmoid()
    
class Tanh(Module):
    def forward(self, x):
        return x.tanh()
    
class GELU(Module):
    def forward(self, x):
        return x.gelu()




# ToDo:
# ReLU       [x]
# LeReLU     [x]
# Sigmoid    [x]
# Tanh       [x]
# GELU       [x]
# kama soska