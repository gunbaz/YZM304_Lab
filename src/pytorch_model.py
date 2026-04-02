"""
YZM304 - Derin Ogrenme | Lab 1
Titanic Binary Classification - PyTorch Modeli
Ankara Universitesi

Yazar : [Adiniz Soyadiniz]
Tarih : 2026-04-02

Mimari (NumPy modeli ile AYNI):
  Giris(22) -> fc1(22->16) -> sigmoid -> fc2(16->1) -> sigmoid
  Loss      : BCELoss (Binary Cross Entropy)
  Optimizer : SGD, lr=0.01
"""

import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# TITANIC NET — nn.Module
# ==============================================================================

class TitanicNet(nn.Module):
    """
    NumPy NeuralNetwork ile birebir ayni mimariye sahip PyTorch modeli.

    Katmanlar:
      fc1 : Linear(22 -> 16)
      fc2 : Linear(16 -> 1)

    Aktivasyon: her katmandan sonra Sigmoid
    """

    def __init__(self, input_size: int = 22, hidden_size: int = 16):
        super(TitanicNet, self).__init__()

        # Tam bagli katmanlar
        self.fc1 = nn.Linear(input_size, hidden_size)  # (22 -> 16)
        self.fc2 = nn.Linear(hidden_size, 1)            # (16 -> 1)

        # Sigmoid aktivasyon
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ileri yayilim.

        x   : (batch_size, 22)
        cikis: (batch_size, 1)  -- [0,1] araliginda olasilik
        """
        x = self.sigmoid(self.fc1(x))   # Gizli katman
        x = self.sigmoid(self.fc2(x))   # Cikis katmani
        return x


# ==============================================================================
# PYTORCH MODEL — egitim ve degerlendirme sarmalayicisi
# ==============================================================================

class PyTorchModel:
    """
    TitanicNet'i egiten ve degerlendiren sarmalayici sinif.

    Kullanim
    --------
    model = PyTorchModel(processed_dir="data/processed")
    model.train()
    model.evaluate()
    model.plot_confusion_matrix(save_path="outputs/")
    model.plot_metrics(save_path="outputs/")
    model.save_results(save_path="outputs/")
    """

    def __init__(self,
                 processed_dir : str   = "data/processed",
                 input_size    : int   = 22,
                 hidden_size   : int   = 16,
                 lr            : float = 0.01,
                 n_epochs      : int   = 10_000,
                 random_state  : int   = 42):

        self.processed_dir = processed_dir
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.lr            = lr
        self.n_epochs      = n_epochs
        self.random_state  = random_state

        # Tekrar ureticilik icin tohum ayarla
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Veriyi yukle ve Tensor'a cevir
        self._load_data()

        # Modeli, kayip fonksiyonunu ve optimizer'i tanimla
        self.net       = TitanicNet(input_size=input_size,
                                    hidden_size=hidden_size)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)

        # Egitim kayiplari
        self.train_losses = []
        self.dev_losses   = []

        # Degerlendirme sonuclari
        self._metrics = {}

    # --------------------------------------------------------------------------
    def _load_data(self):
        """Numpy dizilerini yukle ve PyTorch FloatTensor'a cevir."""
        def _load(fname):
            path = os.path.join(self.processed_dir, fname)
            arr  = np.load(path, allow_pickle=True).astype(np.float32)
            return torch.FloatTensor(arr)

        self.X_train = _load("X_train.npy")
        self.y_train = _load("y_train.npy").unsqueeze(1)   # (N,) -> (N,1)
        self.X_dev   = _load("X_dev.npy")
        self.y_dev   = _load("y_dev.npy").unsqueeze(1)
        self.X_test  = _load("X_test.npy")
        self.y_test  = _load("y_test.npy").unsqueeze(1)

        print(f"[VERI] Train:{tuple(self.X_train.shape)}  "
              f"Dev:{tuple(self.X_dev.shape)}  "
              f"Test:{tuple(self.X_test.shape)}")

    # --------------------------------------------------------------------------
    def _compute_metrics(self, X: torch.Tensor,
                         y_true: torch.Tensor,
                         split_name: str) -> dict:
        """
        Verilen split icin metrikleri hesapla ve dondur.

        Metrikler: accuracy, precision, recall, f1
        """
        self.net.eval()
        with torch.no_grad():
            proba  = self.net(X)                        # (N, 1)
            y_pred = (proba >= 0.5).float()             # 0 veya 1

        # Numpy'a cevir
        y_p = y_pred.numpy().flatten().astype(int)
        y_t = y_true.numpy().flatten().astype(int)

        # El ile hesapla (sklearn bagimliligi olmadan)
        tp = int(np.sum((y_p == 1) & (y_t == 1)))
        fp = int(np.sum((y_p == 1) & (y_t == 0)))
        fn = int(np.sum((y_p == 0) & (y_t == 1)))
        tn = int(np.sum((y_p == 0) & (y_t == 0)))

        acc  = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else 0.0)

        print(f"\n  [{split_name}]")
        print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        return {"accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # ==========================================================================
    # PUBLIC METODLAR
    # ==========================================================================

    def train(self) -> None:
        """
        BCELoss + SGD ile TitanicNet'i egitir.

        Her 1000 epoch'ta train loss ve dev loss yazdirilir.
        Egitim bittikten sonra net.eval() moduna gecilir.
        """
        print("=" * 60)
        print("  PYTORCH TitanicNet EGITIM")
        print(f"  Mimari: {self.input_size} -> {self.hidden_size}(sigmoid) -> 1(sigmoid)")
        print(f"  Loss  : BCELoss | SGD lr={self.lr} | Epochs: {self.n_epochs:,}")
        print("=" * 60)

        self.train_losses = []
        self.dev_losses   = []

        for epoch in range(1, self.n_epochs + 1):
            # --- Egitim modu ---
            self.net.train()
            self.optimizer.zero_grad()

            y_pred     = self.net(self.X_train)          # (N, 1)
            train_loss = self.criterion(y_pred, self.y_train)

            train_loss.backward()
            self.optimizer.step()

            # --- Her 1000 epoch'ta kayit ve ekran ---
            if epoch % 1000 == 0 or epoch == 1:
                self.net.eval()
                with torch.no_grad():
                    dev_pred = self.net(self.X_dev)
                    dev_loss = self.criterion(dev_pred, self.y_dev)

                tl = float(train_loss.item())
                dl = float(dev_loss.item())
                self.train_losses.append((epoch, tl))
                self.dev_losses.append((epoch, dl))

                print(f"  Epoch {epoch:>7,} | "
                      f"Train Loss: {tl:.6f} | "
                      f"Dev Loss: {dl:.6f}")

        # Egitim bitti — eval moduna gec
        self.net.eval()
        print("=" * 60)
        print("  EGITIM TAMAMLANDI")
        print("=" * 60)

    # --------------------------------------------------------------------------
    def evaluate(self) -> None:
        """Train, Dev ve Test setleri icin metrikleri hesapla ve yazdir."""
        print("\n[DEGERLENDIRME] PyTorch TitanicNet")

        self._metrics["train"] = self._compute_metrics(
            self.X_train, self.y_train, "Train")
        self._metrics["dev"]   = self._compute_metrics(
            self.X_dev,   self.y_dev,   "Dev  ")
        self._metrics["test"]  = self._compute_metrics(
            self.X_test,  self.y_test,  "Test ")

        # Overfitting / Underfitting kontrolu
        gap = (self._metrics["train"]["accuracy"] -
               self._metrics["dev"]["accuracy"])
        if gap > 0.10:
            print(f"\n  [!] OVERFITTING riski (Train-Dev farki: {gap:.4f})")
        if self._metrics["train"]["accuracy"] < 0.75:
            print(f"\n  [!] UNDERFITTING riski")

    # --------------------------------------------------------------------------
    def plot_confusion_matrix(self, save_path: str = "outputs/") -> None:
        """
        Test seti icin confusion matrix cizip PNG kaydet.

        Cikti: pytorch_confusion_matrix.png
        """
        if not self._metrics:
            print("[UYARI] Once evaluate() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        m  = self._metrics["test"]
        cm = np.array([[m["tn"], m["fp"]],
                       [m["fn"], m["tp"]]])

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = ["Hayatta Kalmadi (0)", "Hayatta Kaldi (1)"]
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Hucre degerlerini yaz
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        ax.set_ylabel("Gercek Etiket", fontsize=11)
        ax.set_xlabel("Tahmin", fontsize=11)
        ax.set_title("PyTorch TitanicNet — Confusion Matrix (Test)",
                     fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        out = os.path.join(save_path, "pytorch_confusion_matrix.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Confusion matrix -> {out}")

    # --------------------------------------------------------------------------
    def plot_metrics(self, save_path: str = "outputs/") -> None:
        """
        Train / Dev / Test metriklerini ve ogrenme egrisini kaydeder.

        Ciktilar:
          pytorch_metrics.png
          pytorch_learning_curve.png
        """
        if not self._metrics:
            print("[UYARI] Once evaluate() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)

        # --- Bar chart ---
        metric_names = ["accuracy", "precision", "recall", "f1"]
        splits       = ["train", "dev", "test"]
        colors       = ["#4FC3F7", "#A5D6A7", "#EF9A9A"]
        x            = np.arange(len(metric_names))
        width        = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (split, color) in enumerate(zip(splits, colors)):
            vals = [self._metrics[split][m] for m in metric_names]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=split.capitalize(), color=color,
                          edgecolor="white", linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metric_names], fontsize=11)
        ax.set_ylabel("Skor", fontsize=11)
        ax.set_ylim(0.0, 1.15)
        ax.set_title("PyTorch TitanicNet — Train / Dev / Test Metrikler",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        plt.tight_layout()

        out1 = os.path.join(save_path, "pytorch_metrics.png")
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Metrik grafigi -> {out1}")

        # --- Ogrenme egrisi ---
        if self.train_losses:
            steps_tr, losses_tr = zip(*self.train_losses)
            steps_dv, losses_dv = zip(*self.dev_losses)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps_tr, losses_tr, color="#4FC3F7",
                    linewidth=2, label="Train Loss")
            ax.plot(steps_dv, losses_dv, color="#EF9A9A",
                    linewidth=2, linestyle="--", label="Dev Loss")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("BCE Loss", fontsize=12)
            ax.set_title("PyTorch TitanicNet — Ogrenme Egrisi",
                         fontsize=14, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.4)
            plt.tight_layout()

            out2 = os.path.join(save_path, "pytorch_learning_curve.png")
            plt.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[PLOT] Ogrenme egrisi -> {out2}")

    # --------------------------------------------------------------------------
    def save_results(self, save_path: str = "outputs/") -> None:
        """
        Metrikleri pytorch_results.csv dosyasina kaydet.

        Sutunlar: model, split, accuracy, precision, recall, f1
        """
        if not self._metrics:
            print("[UYARI] Once evaluate() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "pytorch_results.csv")
        fields   = ["model", "split", "accuracy", "precision", "recall", "f1"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for split in ["train", "dev", "test"]:
                m = self._metrics[split]
                writer.writerow({
                    "model"    : "PyTorch_TitanicNet",
                    "split"    : split,
                    "accuracy" : round(m["accuracy"],  6),
                    "precision": round(m["precision"], 6),
                    "recall"   : round(m["recall"],    6),
                    "f1"       : round(m["f1"],        6),
                })

        print(f"[CSV] Sonuclar kaydedildi -> {csv_path}")


# ==============================================================================
# MAIN BLOGU
# ==============================================================================
if __name__ == "__main__":
    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")

    model = PyTorchModel(
        processed_dir = PROCESSED_DIR,
        input_size    = 22,
        hidden_size   = 16,
        lr            = 0.01,
        n_epochs      = 10_000,
        random_state  = 42,
    )

    model.train()
    model.evaluate()
    model.plot_confusion_matrix(save_path=OUTPUT_DIR)
    model.plot_metrics(save_path=OUTPUT_DIR)
    model.save_results(save_path=OUTPUT_DIR)

    print("\n[TAMAMLANDI] pytorch_model.py")
