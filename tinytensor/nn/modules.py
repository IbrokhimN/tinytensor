from tinytensor.core.tensor import Tensor
import pickle
import numpy as np
#кароч модуль от которого будут наследоваться все слои
class Module:
    def __init__(self):
        self._modules = {}
        self.training = True

    def to(self, device):
        # переносим все параметры модели либо на cpu либо на cuda
        for name, param in self.__dict__.items():
            if isinstance(param, Tensor):
                setattr(self, name, param.to(device))
            elif isinstance(param, Module):
                param.to(device)
        return self


    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    #щас это просто заглушка для прямого вызова, а в реале у всех должны быть свои реализации forward, так что это роль не играет
    def forward(self, *args, **kwargs):
        raise NotImplementedError(...)
    # сохраняет обучаемые веса, и если в модуле есть еще какие то модули то он их тоже открывает 
    def parameters(self):
        params = []
        for atr in self.__dict__.values():
            if isinstance(atr, Tensor) and atr.requires_grad == True:
                params.append(atr)
            elif isinstance(atr, Module):
                params.extend(atr.parameters())
        return params
    #обнуляет градиенты
    def zero_grad(self):
        for atr in self.parameters():
            atr.grad = None
        
    def quant(self):
        # квантуем все подслои у кого есть свой quant
        self.eval()
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.quant()
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        item.quant()
        return self

    # обучение в стиле keras, если лень писать цикл руками. ручной цикл тоже работает

    def compile(self, optimizer, loss):
        # запоминаем оптимайзер и лосс
        # optimizer можно дать объектом или фабрикой lambda p: AdamW(p)
        if callable(optimizer) and not hasattr(optimizer, "step"):
            # это фабрика, кидаем ей параметры
            optimizer = optimizer(self.parameters())
        self._optimizer = optimizer
        self._loss_fn = loss
        return self

    def fit(self, x, y=None, epochs=1, batch_size=32, shuffle=True,
            verbose=True, validation_data=None, patience=None):
        # обучаем. на вход либо готовый loader, либо сырые x,y
        from tinytensor.data import TensorDataset, DataLoader
        from tinytensor.core.tensor import Tensor
        from tinytensor.utils import EarlyStopping

        if not hasattr(self, "_optimizer") or not hasattr(self, "_loss_fn"):
            raise RuntimeError("Сначала вызови model.compile(optimizer, loss)")

        # если дали loader берем как есть, иначе собираем сами
        if hasattr(x, "__iter__") and y is None and not isinstance(x, (np.ndarray,)):
            loader = x
        else:
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # history чтоб потом график построить
        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        if patience is not None:
            es = EarlyStopping(self, patience=patience)

        self.train()

        for epoch in range(epochs):
            total, n_batches = 0.0, 0

            for xb, yb in loader:
                if not isinstance(xb, Tensor):
                    xb = Tensor(xb)
                if not isinstance(yb, Tensor):
                    yb = Tensor(yb)

                # обычный цикл, просто спрятан внутрь
                pred = self(xb)
                loss = self._loss_fn(pred, yb)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total += float(loss.data)
                n_batches += 1

            avg = total / max(n_batches, 1)
            history["loss"].append(avg)

            line = f"эпоха {epoch+1}/{epochs}  loss={avg:.4f}"
            # валидация если дали
            if validation_data is not None:
                val_loss = self.evaluate(*validation_data, batch_size=batch_size, verbose=False)
                history["val_loss"].append(val_loss)
                line += f"  val_loss={val_loss:.4f}"
                if patience is not None:
                    if es(val_loss):
                        print("ранняя остановка")
                        break

            if verbose:
                print(line)

        return history

    def evaluate(self, x, y=None, batch_size=32, verbose=True):
        # просто считаем лосс без обучения
        from tinytensor.data import TensorDataset, DataLoader
        from tinytensor.core.tensor import Tensor

        if hasattr(x, "__iter__") and y is None and not isinstance(x, (np.ndarray,)):
            loader = x
        else:
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        total, n_batches = 0.0, 0
        for xb, yb in loader:
            if not isinstance(xb, Tensor):
                xb = Tensor(xb)
            if not isinstance(yb, Tensor):
                yb = Tensor(yb)
            pred = self(xb)
            loss = self._loss_fn(pred, yb)
            total += float(loss.data)
            n_batches += 1

        avg = total / max(n_batches, 1)
        if verbose:
            print(f"evaluate  loss={avg:.4f}")
        return avg

    def train(self, mode=True):
        self.training = mode
        for atr in self.__dict__.values():
            if isinstance(atr, Module):
                atr.train(mode)
        return self
    def eval(self):
        return self.train(False)
    def state_dict(self):
        #вытащим все веса параметров
        sdict = {}
        for name, param in self._get_named_params().items():
            sdict[name] = param.data
        # плюс int8 буферы у квантованных слоёв
        for name, buffers in self._get_named_buffers().items():
            for bname, bval in buffers.items():
                sdict[f"{name}.{bname}"] = bval
        return sdict

    def _get_named_buffers(self, prefix=""):
        # собираем int8 буферы по всему дереву
        named = {}
        if getattr(self, "quantized", False) and hasattr(self, "_quant_buffers"):
            named[prefix if prefix else "_self"] = self._quant_buffers()
        for attr_name, attr in self.__dict__.items():
            key = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(attr, Module):
                named.update(attr._get_named_buffers(prefix=key))
            elif isinstance(attr, list):
                for i, item in enumerate(attr):
                    if isinstance(item, Module):
                        named.update(item._get_named_buffers(prefix=f"{key}.{i}"))
        return named           
        
    def load_state_dict(self, sdict):
        #запихиваем обратно float-параметры
        for name, param in self._get_named_params().items():
            if name in sdict:
                param.data = sdict[name]
        # и int8 буферы если есть
        self._load_quant_buffers(sdict)

    def _load_quant_buffers(self, sdict, prefix=""):
        # грузим int8 буферы и включаем квант-режим
        if hasattr(self, "_set_quant_buffers"):
            self._set_quant_buffers(sdict, prefix)
        for attr_name, attr in self.__dict__.items():
            key = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(attr, Module):
                attr._load_quant_buffers(sdict, prefix=key)
            elif isinstance(attr, list):
                for i, item in enumerate(attr):
                    if isinstance(item, Module):
                        item._load_quant_buffers(sdict, prefix=f"{key}.{i}")
    def _get_named_params(self, prefix=""):
        named = {}
        for attr_name, attr in self.__dict__.items():
            key = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(attr, Tensor):
                named[key] = attr
            elif isinstance(attr, Module):
                named.update(attr._get_named_params(prefix=key))
        return named 
    
    #сохранение модели
    def save(self,filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.state_dict(), f)
    # загрузка весов обратно
    def load(self, filepath):
        with open(filepath, "rb") as f:
            sd = pickle.load(f)
            self.load_state_dict(sd)
        

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self

    def _get_named_params(self, prefix=""):
        named = {}
        for i, layer in enumerate(self.layers):
            key = f"{prefix}.layers.{i}" if prefix else f"layers.{i}"
            named.update(layer._get_named_params(prefix=key))
        return named

    def forward(self, x):
        # закидываем x по слоям
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def prune(self, amount=0.3):
        from tinytensor.nn.linear import Linear
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Linear):
                continue
            nxt = None
            for j in range(i + 1, len(self.layers)):
                if isinstance(self.layers[j], Linear):
                    nxt = self.layers[j]
                    break

            if nxt is None:
                continue
            w = layer.weight.data
            importance = np.linalg.norm(w, axis=0)
            deleting = int(layer.out_features * amount)
            if deleting == 0:
                continue

            idx = np.argsort(importance)[:deleting]
            layer._prune_outputs(idx)
            nxt._prune_inputs(idx)

        return self

    def to_onnx(self, path, input_dim):
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        from tinytensor.nn.linear import Linear
        from tinytensor.nn.activations import ReLU
        nodes = []
        initializers = []
        cur = "input"
        for i, layer in enumerate(self.layers):
            #Gemm
            if isinstance(layer, Linear):
                w_init = numpy_helper.from_array(layer.weight.data, name=f"W{i}")
                b_init = numpy_helper.from_array(layer.bias.data, name=f"B{i}")
                initializers.extend([w_init, b_init])

                node = helper.make_node("Gemm", inputs=[cur, f"W{i}", f"B{i}"], outputs=[f"h{i}"])
                nodes.append(node)
                cur = f"h{i}"

            #Relu
            elif isinstance(layer, ReLU):
                node = helper.make_node("Relu", inputs=[cur], outputs=[f"h{i}"])
                nodes.append(node)
                cur = f"h{i}"
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", input_dim])
        out = helper.make_tensor_value_info(cur, TensorProto.FLOAT, ["N", self.layers[-1].out_features])
        graph = helper.make_graph(nodes, "model", [inp], [out], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save(model, path)

