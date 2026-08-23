# dict с методом .plot(), доступ по ключам как обычно

class History(dict):
    def plot(self, save_path=None):
        # matplotlib не в зависимостях, грузим лениво
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Для history.plot() нужен matplotlib. Установи его:")
            print("    pip install matplotlib")
            return

        has_acc = "acc" in self and len(self["acc"]) > 0
        n_plots = 2 if has_acc else 1

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        ax = axes[0]
        epochs = range(1, len(self["loss"]) + 1)
        ax.plot(epochs, self["loss"], label="train loss")
        if "val_loss" in self and len(self["val_loss"]) > 0:
            val_epochs = range(1, len(self["val_loss"]) + 1)
            ax.plot(val_epochs, self["val_loss"], label="val loss")
        ax.set_xlabel("эпоха")
        ax.set_ylabel("loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if has_acc:
            ax = axes[1]
            acc_epochs = range(1, len(self["acc"]) + 1)
            ax.plot(acc_epochs, self["acc"], label="train acc", color="green")
            ax.set_xlabel("эпоха")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=120)
            print(f"график сохранён: {save_path}")
        else:
            plt.show()
