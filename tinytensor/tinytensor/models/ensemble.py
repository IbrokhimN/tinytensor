import numpy as np


class Ensemble:
    # ансамбль моделей. внутрь можно кидать готовые архитектуры (LeNet, ResNet...)
    # и свои Sequential. обучаются по очереди, на инференсе голосуют.
    def __init__(self, models, voting_type="soft"):
        # models - список моделей (любой Module)
        # voting_type - "soft" (по вероятностям) или "hard" (по классам)
        if voting_type not in ("soft", "hard"):
            raise ValueError("voting_type должен быть 'soft' или 'hard'")
        self.models = list(models)
        self.voting_type = voting_type

    def compile(self, optimizer, loss):
        # компилируем все модели одинаково.
        # optimizer лучше давать фабрикой lambda p: AdamW(p), чтобы у каждой
        # модели был свой оптимайзер со своими параметрами
        for m in self.models:
            m.compile(optimizer, loss)
        return self

    def fit(self, x, y=None, **kwargs):
        # обучаем по очереди: сначала модель 1, потом 2, и т.д.
        histories = []
        for i, m in enumerate(self.models):
            print(f"=== обучение модели {i+1}/{len(self.models)} ({type(m).__name__}) ===")
            h = m.fit(x, y, **kwargs)
            histories.append(h)
        self.histories = histories   # запоминаем для plot
        return histories

    def predict_proba(self, x, batch_size=32):
        # усреднённые вероятности по всем моделям, форма (N, n_classes)
        total = None
        for m in self.models:
            probs = m.predict_proba(x, batch_size=batch_size)
            total = probs if total is None else total + probs
        return total / len(self.models)

    def predict(self, x, batch_size=32):
        # предсказанный класс на объект, форма (N,)
        if self.voting_type == "soft":
            # суммируем вероятности всех моделей, берём argmax суммы
            return np.argmax(self.predict_proba(x, batch_size=batch_size), axis=1)
        else:
            # hard: каждая модель даёт свой класс, берём самый частый (мода)
            votes = []
            for m in self.models:
                votes.append(m.predict(x, batch_size=batch_size))
            votes = np.stack(votes, axis=0)   # (n_models, N)
            # для каждого объекта считаем какой класс встретился чаще всего
            n = votes.shape[1]
            out = np.empty(n, dtype=votes.dtype)
            for j in range(n):
                counts = np.bincount(votes[:, j])
                out[j] = np.argmax(counts)
            return out

    def evaluate(self, x, y, batch_size=32):
        # доля правильных ответов ансамбля на тесте
        preds = self.predict(x, batch_size=batch_size)
        return float((preds == y).mean())

    def _model_accuracy(self, model, x, y, batch_size=32):
        # тестовая точность одной модели
        preds = model.predict(x, batch_size=batch_size)
        return float((preds == y).mean())

    def plot_comparison(self, histories, x_test, y_test, batch_size=32, save_path=None):
        # график: train-accuracy каждой модели по эпохам + финальная тестовая
        # точность каждой модели (точки) и линия ансамбля (сверху).
        # histories - список, который вернул ens.fit(...)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Для plot_comparison нужен matplotlib. Установи его:")
            print("    pip install matplotlib")
            return

        names = [type(m).__name__ for m in self.models]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # кривые train-accuracy по эпохам
        for i, (h, name) in enumerate(zip(histories, names)):
            if "acc" in h and len(h["acc"]) > 0:
                epochs = range(1, len(h["acc"]) + 1)
                acc_pct = [a * 100 for a in h["acc"]]
                ax.plot(epochs, acc_pct, marker="o", color=colors[i % len(colors)],
                        label=f"{name} (train)")

        # финальная тестовая точность каждой модели (точки на последней эпохе)
        last_epoch = max((len(h.get("acc", [])) for h in histories), default=1)
        for i, (m, name) in enumerate(zip(self.models, names)):
            test_acc = self._model_accuracy(m, x_test, y_test, batch_size) * 100
            ax.scatter([last_epoch], [test_acc], color=colors[i % len(colors)],
                       marker="*", s=200, zorder=5,
                       label=f"{name} (test {test_acc:.2f}%)")

        # линия ансамбля сверху
        ens_acc = self.evaluate(x_test, y_test, batch_size) * 100
        ax.axhline(ens_acc, color="red", linestyle="--", linewidth=2,
                   label=f"Ансамбль (test {ens_acc:.2f}%)")

        ax.set_xlabel("эпоха")
        ax.set_ylabel("accuracy, %")
        ax.set_title(f"Модели vs ансамбль ({self.voting_type} voting)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=120)
            print(f"график сохранён: {save_path}")
        else:
            plt.show()

    def __len__(self):
        return len(self.models)

    def save(self, filepath):
        # сохраняем веса всех моделей + voting_type в один файл
        import pickle
        blob = {
            "voting_type": self.voting_type,
            "states": [m.state_dict() for m in self.models],
        }
        with open(filepath, "wb") as f:
            pickle.dump(blob, f)

    def load(self, filepath):
        # грузим обратно. модели в ансамбле должны быть той же архитектуры
        # и в том же порядке, что при сохранении (load не создаёт слои сам)
        import pickle
        with open(filepath, "rb") as f:
            blob = pickle.load(f)
        if len(blob["states"]) != len(self.models):
            raise ValueError(
                f"в файле {len(blob['states'])} моделей, а в ансамбле {len(self.models)}"
            )
        self.voting_type = blob["voting_type"]
        for m, sd in zip(self.models, blob["states"]):
            m.load_state_dict(sd)
        return self

    def plot(self, x_test, y_test, save_path=None, batch_size=32):
        # график val-accuracy каждой модели по эпохам + честная точность ансамбля.
        # x_test, y_test нужны чтобы посчитать точность ансамбля на тесте.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Для plot() нужен matplotlib. Установи его:")
            print("    pip install matplotlib")
            return

        if not hasattr(self, "histories"):
            print("сначала обучи ансамбль через fit(...)")
            return

        # проверяем что была валидация (иначе val_acc нет)
        if "val_acc" not in self.histories[0]:
            print("val_acc нет в истории. Обучай с validation_data=(x_val, y_val), "
                  "иначе честной точности по эпохам не будет.")
            return

        plt.figure(figsize=(8, 5))

        # линия val_acc на каждую модель
        for m, h in zip(self.models, self.histories):
            accs = [a * 100 for a in h["val_acc"]]
            epochs = range(1, len(accs) + 1)
            plt.plot(epochs, accs, marker="o", label=f"{type(m).__name__} (val)")

        # честная точность ансамбля на тесте - одно число, рисуем линией
        ens_acc = self.evaluate(x_test, y_test, batch_size=batch_size) * 100
        plt.axhline(ens_acc, color="red", linestyle="--", linewidth=2,
                    label=f"Ансамбль ({self.voting_type}): {ens_acc:.2f}%")

        plt.xlabel("эпоха")
        plt.ylabel("accuracy, %")
        plt.title("Val-accuracy моделей vs ансамбль")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=120)
            print(f"график сохранён: {save_path}")
        else:
            plt.show()

        return ens_acc
