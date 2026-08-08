import numpy as np
from tinytensor.core.tensor import Tensor
from tinytensor.nn.modules import Module 

def summary(model: Module, input_shape: tuple):
    # печает инфу о модели
    print("\n" + "=" * 50)
    print(f"{'layer (type)':<20} {'output shape':<18} {'param #':<10}")
    print("-" * 50)

    total_params = 0
    trainable_params = 0

    dummy_input = Tensor(np.zeros(input_shape), requires_grad=False)
    x = dummy_input

    layers = []
    for attr in model.__dict__.values():
        if isinstance(attr, Module):
            layers.append(attr)
        elif isinstance(attr, (list, tuple)):
            for item in attr:
                if isinstance(item, Module):
                    layers.append(item)

    for layer in layers:
        layer_name = layer.__class__.__name__

        layer_params = 0
        for p in layer.parameters():
            num_params = int(np.prod(p.data.shape))
            layer_params += num_params
            if p.requires_grad:
                trainable_params += num_params
            total_params += num_params

        try:
            x = layer(x)
            out_shape = str(tuple(x.data.shape))
        except Exception as e:
            out_shape = "Ошибка"

        print(f"{layer_name:<20} {out_shape:<18} {layer_params:<10,}")

    print("-" * 50)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    # размер модели во float32 (4 байта на параметр)
    print(f"Size (float32): {total_params * 4 / (1024 ** 2):.3f} MB")
    print("=" * 50 + "\n")
