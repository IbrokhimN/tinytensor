# tinytensor.models

Ready-made model architectures, assembled from `tinytensor.nn` layers. Each is a
`Module` you can train directly with `compile`/`fit`.

```python
from tinytensor.models import LeNet, VGG, ResNet18, ResNet34
```

- [LeNet](#lenet)
- [VGG](#vgg)
- [ResNet](#resnet)

## LeNet

Classic CNN (LeCun, 1998): two conv+pool blocks then three fully-connected
layers. Sized for 28x28 inputs (MNIST). About 62k parameters. Light enough to
train fully on CPU.

```python
LeNet(num_classes=10, in_channels=1)
```

**Args:**

- `num_classes` (int): number of output classes.
- `in_channels` (int): input channels (1 for grayscale, 3 for RGB).

```python
model = LeNet(num_classes=10, in_channels=1)
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(x_train, y_train, epochs=5)
```

## VGG

VGG-style network: stacks of 3x3 convolutions grouped into blocks, each block
followed by pooling (halving spatial size, doubling channels). Sized for 32x32
inputs. About 3.8M parameters.

```python
VGG(num_classes=10, in_channels=3, small_input=True)
```

**Args:**

- `num_classes` (int): number of output classes.
- `in_channels` (int): input channels.
- `small_input` (bool): tuned for 32x32 images.

## ResNet

Residual network built from `ResidualBlock`. Each stage doubles channels and
halves spatial size; the first block of a stage handles the shape change with a
1x1 downsample on the skip path.

```python
ResNet(num_classes=10, in_channels=3, blocks_per_stage=(2, 2, 2, 2), small_input=True)
```

**Args:**

- `num_classes` (int): number of output classes.
- `in_channels` (int): input channels.
- `blocks_per_stage` (tuple): number of residual blocks in each of the 4 stages.
  `(2,2,2,2)` is ResNet-18, `(3,4,6,3)` is ResNet-34.
- `small_input` (bool): if `True`, uses a light 3x3 stem for small images
  (CIFAR/MNIST); if `False`, the classic 7x7 stem + maxpool for 224px images.

**Convenience factories:**

```python
ResNet18(num_classes=10, in_channels=3, small_input=True)   # (2,2,2,2)
ResNet34(num_classes=10, in_channels=3, small_input=True)   # (3,4,6,3)
```

```python
model = ResNet18(num_classes=10, in_channels=1, small_input=True)
```

> **Note:** ResNet18 is ~11M parameters. On the NumPy/CuPy backend it is heavy —
> train it on GPU (`tt.set_device("cuda")`); on CPU it is very slow and may run
> out of memory. For CPU training, prefer `LeNet`.
