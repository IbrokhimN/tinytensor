from tinytensor.nn.modules import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def _get_named_params(self, prefix=""):
        named = {}
        for i, layer in enumerate(self.layers):
            key = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
            named.update(layer._get_named_params(prefix=key))
        return named

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def train(self, mode=True):
        super().train(mode)
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train(mode)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
