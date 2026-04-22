"""
YZM304 - Derin Ogrenme | Lab 1
Titanic Binary Classification - Scikit-learn MLPClassifier
Ankara Universitesi

Yazar : [Adiniz Soyadiniz]
Tarih : 2026-04-02

Mimari (NumPy modeli ile AYNI):
  Giris(22) -> Gizli(16, sigmoid) -> Cikis(1, sigmoid)
  Loss   : log_loss (Binary Cross Entropy)
  Solver : sgd
"""

import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              ConfusionMatrixDisplay)


class SklearnModel:
    """
    Scikit-learn MLPClassifier ile Titanic ikili siniflandirmasi.

    Mimari NumPy modeliyle eslestirilmistir:
      - 1 gizli katman, 16 noron
      - sigmoid aktivasyon (logistic)
      - SGD optimizer, lr=0.01, max_iter=10000

    Kullanim
    --------
    model = SklearnModel(processed_dir="data/processed")
    model.train()
    model.evaluate()
    model.plot_confusion_matrix(save_path="outputs/")
    model.plot_metrics(save_path="outputs/")
    model.save_results(save_path="outputs/")
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

        # Veriyi yukle
        self._load_data()

        # MLPClassifier tanimla — NumPy modeliyle ayni mimari
        self.clf = MLPClassifier(
            hidden_layer_sizes=(16,),      # 1 gizli katman, 16 noron
            activation="logistic",          # sigmoid
            solver="sgd",                   # Stochastic Gradient Descent
            learning_rate_init=0.01,        # lr=0.01
            max_iter=10_000,                # n_steps ile eslestirildi
            random_state=42,
            verbose=False,
            early_stopping=False,
            n_iter_no_change=10_000,        # erken durmayi devre disi birak
        )

        # Degerlendirme sonuclari
        self._metrics = {}

    # --------------------------------------------------------------------------
    def _load_data(self):
        """data/processed/ klasöründen verileri yukle."""
        def _load(fname):
            path = os.path.join(self.processed_dir, fname)
            arr  = np.load(path, allow_pickle=True)
            return arr.astype(np.float64)

        self.X_train = _load("X_train.npy")
        self.y_train = _load("y_train.npy").astype(int)
        self.X_dev   = _load("X_dev.npy")
        self.y_dev   = _load("y_dev.npy").astype(int)
        self.X_test  = _load("X_test.npy")
        self.y_test  = _load("y_test.npy").astype(int)

        print(f"[VERI] Train:{self.X_train.shape}  "
              f"Dev:{self.X_dev.shape}  Test:{self.X_test.shape}")

    # --------------------------------------------------------------------------
    def _compute_metrics(self, X, y_true, split_name: str) -> dict:
        """
        Verilen split icin dört metrik hesapla ve dondur.

        Metrikler: accuracy, precision, recall, f1
        """
        y_pred = self.clf.predict(X)
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n  [{split_name}]")
        print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # ==========================================================================
    # PUBLIC METODLAR
    # ==========================================================================

    def train(self) -> None:
        """
        MLPClassifier'i egitim verisiyle egitir.
        Sklearn icsel olarak tum iterasyonlari yonetir.
        """
        print("=" * 60)
        print("  SKLEARN MLPClassifier EGITIM")
        print("  Mimari: 22 -> 16(sigmoid) -> 1(sigmoid)")
        print("  Solver: sgd | LR: 0.01 | MaxIter: 10,000")
        print("=" * 60)

        # Egitim — sklearn hem X_train hem y_train'i kullanir
        self.clf.fit(self.X_train, self.y_train)

        print(f"  Egitim tamamlandi. "
              f"Gercek iterasyon sayisi: {self.clf.n_iter_}")
        print(f"  Son loss: {self.clf.loss_:.6f}")
        print("=" * 60)

    # --------------------------------------------------------------------------
    def evaluate(self) -> None:
        """Train, Dev ve Test setleri icin metrikleri hesaplayip yazdirir."""
        print("\n[DEGERLENDIRME] Sklearn MLPClassifier")

        self._metrics["train"] = self._compute_metrics(
            self.X_train, self.y_train, "Train")
        self._metrics["dev"]   = self._compute_metrics(
            self.X_dev,   self.y_dev,   "Dev  ")
        self._metrics["test"]  = self._compute_metrics(
            self.X_test,  self.y_test,  "Test ")

        # Overfitting / underfitting kontrolu
        gap = (self._metrics["train"]["accuracy"] -
               self._metrics["dev"]["accuracy"])
        if gap > 0.10:
            print(f"\n  [!] OVERFITTING riski (Train-Dev farki: {gap:.4f})")
        if self._metrics["train"]["accuracy"] < 0.75:
            print(f"\n  [!] UNDERFITTING riski")

    # --------------------------------------------------------------------------
    def plot_confusion_matrix(self, save_path: str = "outputs/") -> None:
        """
        Test seti icin confusion matrix gorsellestir ve PNG kaydet.

        Cikti: sklearn_confusion_matrix.png
        """
        os.makedirs(save_path, exist_ok=True)
        y_pred = self.clf.predict(self.X_test)
        cm     = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Hayatta Kalmadi (0)", "Hayatta Kaldi (1)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Sklearn MLP — Confusion Matrix (Test)",
                     fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        out = os.path.join(save_path, "sklearn_confusion_matrix.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Confusion matrix -> {out}")

    # --------------------------------------------------------------------------
    def plot_metrics(self, save_path: str = "outputs/") -> None:
        """
        Train / Dev / Test metriklerini grouped bar chart ile gorsellestir.

        Cikti: sklearn_metrics.png
        """
        if not self._metrics:
            print("[UYARI] Once evaluate() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        metric_names = ["accuracy", "precision", "recall", "f1"]
        splits       = ["train", "dev", "test"]
        colors       = ["#4FC3F7", "#A5D6A7", "#EF9A9A"]

        x     = np.arange(len(metric_names))
        width = 0.25

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
        ax.set_title("Sklearn MLP — Train / Dev / Test Metrikler",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.35)
        plt.tight_layout()

        out = os.path.join(save_path, "sklearn_metrics.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Metrik grafigi -> {out}")

    # --------------------------------------------------------------------------
    def save_results(self, save_path: str = "outputs/") -> None:
        """
        Metrikleri sklearn_results.csv dosyasina kaydet.

        Sutunlar: model, split, accuracy, precision, recall, f1
        """
        if not self._metrics:
            print("[UYARI] Once evaluate() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "sklearn_results.csv")
        fields   = ["model", "split", "accuracy", "precision", "recall", "f1"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for split in ["train", "dev", "test"]:
                m = self._metrics[split]
                writer.writerow({
                    "model"    : "Sklearn_MLP",
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

    model = SklearnModel(processed_dir=PROCESSED_DIR)

    model.train()
    model.evaluate()
    model.plot_confusion_matrix(save_path=OUTPUT_DIR)
    model.plot_metrics(save_path=OUTPUT_DIR)
    model.save_results(save_path=OUTPUT_DIR)

    print("\n[TAMAMLANDI] sklearn_model.py")
