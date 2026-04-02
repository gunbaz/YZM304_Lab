"""
YZM304 - Derin Öğrenme | Lab 1
Titanic Binary Classification - Model Egitimi ve Karsilastirma
Ankara Universitesi

Yazar : [Adiniz Soyadiniz]
Tarih : 2026-04-02

5 model egitilip karsilastirilir:
  Model 1 - Baseline      : hidden=16,  2 katman
  Model 2 - Derin         : hidden=16,  3 katman (DeepNeuralNetwork)
  Model 3 - Genis         : hidden=64,  2 katman
  Model 4 - L2 Reg        : hidden=16,  2 katman + L2 regularization
  Model 5 - Dropout       : hidden=32,  2 katman + Dropout
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Ayni src/ klasoründen neural_network.py'i import et ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neural_network import NeuralNetwork


# ==============================================================================
# YARDIMCI MODEL SINIFLARI  (NeuralNetwork'u extend eder)
# ==============================================================================

class DeepNeuralNetwork(NeuralNetwork):
    """
    Uc gizli katmanli sinir agi.

    Mimari: giris(input_size) -> gizli1(hidden_size, sigmoid)
                              -> gizli2(hidden_size, sigmoid)
                              -> cikis(1, sigmoid)

    NeuralNetwork'e ek olarak W3, b3 eklendi.
    _initialize_weights, _forward, _backward, _update_weights override edildi.
    """

    def __init__(self, input_size, hidden_size=16,
                 learning_rate=0.01, n_steps=10_000, random_state=42):
        # Ust sinifin __init__'ini cagir; bu _initialize_weights'i da tetikler
        super().__init__(input_size, hidden_size, learning_rate,
                         n_steps, random_state)

    # --- Ek agirliklar dahil yeniden baslatma ---
    def _initialize_weights(self):
        np.random.seed(self.random_state)

        # Gizli katman 1: W1 (hidden x input)
        std1 = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * std1
        self.b1 = np.zeros((self.hidden_size, 1))

        # Gizli katman 2: W2 (hidden x hidden)  <- yeni katman
        std2 = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size) * std2
        self.b2 = np.zeros((self.hidden_size, 1))

        # Cikis katmani: W3 (1 x hidden)
        std3 = np.sqrt(2.0 / (self.hidden_size + 1))
        self.W3 = np.random.randn(1, self.hidden_size) * std3
        self.b3 = np.zeros((1, 1))

    def _forward(self, X):
        """3 katmanli ileri yayilim."""
        X_T = X.T

        # Katman 1
        Z1 = self.W1 @ X_T + self.b1
        A1 = self._sigmoid(Z1)

        # Katman 2 (ek gizli katman)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self._sigmoid(Z2)

        # Cikis katmani
        Z3 = self.W3 @ A2 + self.b3
        A3 = self._sigmoid(Z3)

        # Backward icin onbellege al
        self._cache = {
            "X" : X,
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2,
            "Z3": Z3, "A3": A3,
        }
        return A3.T   # (n_samples, 1)

    def _backward(self, X, y):
        """3 katmanli geri yayilim."""
        A1 = self._cache["A1"]
        A2 = self._cache["A2"]
        A3 = self._cache["A3"]
        n  = X.shape[0]
        y_T = y.T   # (1, n_samples)

        # Cikis katmani
        dZ3 = A3 - y_T
        dW3 = (1.0 / n) * (dZ3 @ A2.T)
        db3 = (1.0 / n) * np.sum(dZ3, axis=1, keepdims=True)

        # Gizli katman 2
        dA2 = self.W3.T @ dZ3
        dZ2 = dA2 * A2 * (1.0 - A2)
        dW2 = (1.0 / n) * (dZ2 @ A1.T)
        db2 = (1.0 / n) * np.sum(dZ2, axis=1, keepdims=True)

        # Gizli katman 1
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * A1 * (1.0 - A1)
        dW1 = (1.0 / n) * (dZ1 @ X)
        db1 = (1.0 / n) * np.sum(dZ1, axis=1, keepdims=True)

        self._grads = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3,
        }

    def _update_weights(self):
        """3 katmanin agirlkllarini SGD ile guncelle."""
        self.W1 -= self.lr * self._grads["dW1"]
        self.b1 -= self.lr * self._grads["db1"]
        self.W2 -= self.lr * self._grads["dW2"]
        self.b2 -= self.lr * self._grads["db2"]
        self.W3 -= self.lr * self._grads["dW3"]
        self.b3 -= self.lr * self._grads["db3"]

    def save_weights(self, path="outputs/model_weights.npz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
        print(f"[SAVE] Agirliklar kaydedildi -> {path}")


# ------------------------------------------------------------------------------

class RegularizedNN(NeuralNetwork):
    """
    L2 regularization (weight decay) eklenmiş sinir agi.

    Loss = BCE + lambda_l2 * (||W1||^2 + ||W2||^2)
    Gradyan guncellemesinde her agirliga  -lr * lambda_l2 * W  eklenir.
    """

    def __init__(self, input_size, hidden_size=16,
                 learning_rate=0.01, n_steps=10_000,
                 random_state=42, lambda_l2=0.01):
        self.lambda_l2 = lambda_l2
        super().__init__(input_size, hidden_size, learning_rate,
                         n_steps, random_state)

    def _compute_loss(self, y_pred, y_true):
        """BCE + L2 regularization terimi."""
        bce = super()._compute_loss(y_pred, y_true)
        # L2 cezasi: lambda/2 * (||W1||^2 + ||W2||^2)
        l2_penalty = (self.lambda_l2 / 2.0) * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
        )
        return bce + l2_penalty

    def _update_weights(self):
        """
        SGD + L2 weight decay:
        W <- W - lr * (dW + lambda * W)
            =  W * (1 - lr*lambda)  - lr * dW
        """
        # L2 decay terimi agirlik guncellemesine eklenir
        self.W1 -= self.lr * (self._grads["dW1"] + self.lambda_l2 * self.W1)
        self.b1 -= self.lr * self._grads["db1"]
        self.W2 -= self.lr * (self._grads["dW2"] + self.lambda_l2 * self.W2)
        self.b2 -= self.lr * self._grads["db2"]


# ------------------------------------------------------------------------------

class DropoutNN(NeuralNetwork):
    """
    Gizli katmana Dropout regularization eklenmiş sinir agi.

    Egitim sirasinda: her adimda gizli nöronlarin dropout_rate kadarini maskele.
    Tahminde: Dropout KAPALI (tum noronlari kullan, ciktiyi olcekle).
    """

    def __init__(self, input_size, hidden_size=32,
                 learning_rate=0.01, n_steps=10_000,
                 random_state=42, dropout_rate=0.3):
        self.dropout_rate     = dropout_rate
        self._is_training     = False   # fit() icinde True yapilir
        self._dropout_mask    = None
        super().__init__(input_size, hidden_size, learning_rate,
                         n_steps, random_state)

    def _forward(self, X):
        """
        Egitimde  : Dropout maskelesi uygula (inverted dropout).
        Tahmin de : Maskeleme yok, tam degerlerle hesapla.
        """
        X_T = X.T

        # Gizli katman
        Z1 = self.W1 @ X_T + self.b1
        A1 = self._sigmoid(Z1)

        if self._is_training:
            # Inverted Dropout: Maskeleme + olcekleme (test sirasinda duzeltme gereksiz)
            self._dropout_mask = (
                np.random.rand(*A1.shape) > self.dropout_rate
            ).astype(np.float64)
            A1 = (A1 * self._dropout_mask) / (1.0 - self.dropout_rate + 1e-8)

        # Cikis katmani
        Z2 = self.W2 @ A1 + self.b2
        A2 = self._sigmoid(Z2)

        self._cache = {
            "X" : X,
            "Z1": Z1, "A1": A1,   # A1 zaten maskelenmis (egitimde)
            "Z2": Z2, "A2": A2,
        }
        return A2.T

    def _backward(self, X, y):
        """Dropout maskesi geriye dogru da uygulanir."""
        A1 = self._cache["A1"]
        A2 = self._cache["A2"]
        n  = X.shape[0]
        y_T = y.T

        # Cikis gradyani
        dZ2 = A2 - y_T
        dW2 = (1.0 / n) * (dZ2 @ A1.T)
        db2 = (1.0 / n) * np.sum(dZ2, axis=1, keepdims=True)

        # Gizli katman gradyani (dropout maskesi geri yayilimda da uygulanir)
        dA1 = self.W2.T @ dZ2
        if self._is_training and self._dropout_mask is not None:
            dA1 = (dA1 * self._dropout_mask) / (1.0 - self.dropout_rate + 1e-8)

        dZ1 = dA1 * A1 * (1.0 - A1)
        dW1 = (1.0 / n) * (dZ1 @ X)
        db1 = (1.0 / n) * np.sum(dZ1, axis=1, keepdims=True)

        self._grads = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
        }

    def fit(self, X, y, X_dev=None, y_dev=None):
        """Egitimi dropout ACIK baslat, bitince kapat."""
        self._is_training = True
        super().fit(X, y, X_dev=X_dev, y_dev=y_dev)
        self._is_training = False   # tahmin icin dropout kapat

    def predict_proba(self, X):
        """Tahmin: dropout kapali."""
        self._is_training = False
        return super().predict_proba(X)


# ==============================================================================
# MODEL TRAINER
# ==============================================================================

class ModelTrainer:
    """
    5 farkli sinir agi modelini egitir ve karsilastirir.

    Kullanim
    --------
    trainer = ModelTrainer(processed_dir="data/processed",
                           output_dir="outputs")
    trainer.train_all_models()
    trainer.compare_models()
    trainer.select_best_model(accuracy_threshold=0.80)
    trainer.plot_comparison(save_path="outputs/")
    trainer.save_results(save_path="outputs/")
    """

    # Karsilastirma renk paleti — her model icin farkli renk
    _COLORS = ["#4FC3F7", "#EF9A9A", "#A5D6A7", "#FFE082", "#CE93D8"]

    def __init__(self,
                 processed_dir: str = "data/processed",
                 output_dir:    str = "outputs"):
        self.processed_dir = processed_dir
        self.output_dir    = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Veriyi otomatik yukle
        self._load_data()

        # Egitim sonuclari: {model_adi: {"model": ..., "train_acc": ..., ...}}
        self._results = {}

    # ──────────────────────────────────────────────────────────────────────────
    def _load_data(self):
        """data/processed/ klasoründen tum npy dosyalarini yukle."""
        def _load(filename):
            path = os.path.join(self.processed_dir, filename)
            arr  = np.load(path, allow_pickle=True)
            return arr.astype(np.float64)

        self.X_train = _load("X_train.npy")
        self.y_train = _load("y_train.npy")
        self.X_dev   = _load("X_dev.npy")
        self.y_dev   = _load("y_dev.npy")
        self.X_test  = _load("X_test.npy")
        self.y_test  = _load("y_test.npy")

        n_feats = self.X_train.shape[1]
        print(f"[VERI] Train:{self.X_train.shape}  Dev:{self.X_dev.shape}"
              f"  Test:{self.X_test.shape}  Ozellik:{n_feats}")

    # ──────────────────────────────────────────────────────────────────────────
    def _detect_fitting(self, model_name, train_acc, dev_acc):
        """
        Overfitting / Underfitting tespiti yapar ve uyari yazdirir.

        Kural:
          - |train_acc - dev_acc| > 0.10  => Overfitting riski
          - train_acc < 0.75              => Underfitting riski
        """
        gap = train_acc - dev_acc
        if gap > 0.10:
            print(f"  [!] {model_name}: OVERFITTING riski "
                  f"(Train-Dev farki: {gap:.4f})")
        if train_acc < 0.75:
            print(f"  [!] {model_name}: UNDERFITTING riski "
                  f"(Train accuracy: {train_acc:.4f})")
        if gap <= 0.10 and train_acc >= 0.75:
            print(f"  [OK] {model_name}: Iyi genelleme")

    # ──────────────────────────────────────────────────────────────────────────
    def _train_single(self, name, model):
        """Tek bir modeli egit, degerlendir ve sonucu sakla."""
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Egitim
        model.fit(self.X_train, self.y_train,
                  X_dev=self.X_dev, y_dev=self.y_dev)

        # Degerlendirme
        tr = model.evaluate(self.X_train, self.y_train, set_name="Train")
        dv = model.evaluate(self.X_dev,   self.y_dev,   set_name="Dev  ")
        te = model.evaluate(self.X_test,  self.y_test,  set_name="Test ")

        # Overfitting / Underfitting kontrolu
        self._detect_fitting(name, tr["accuracy"], dv["accuracy"])

        # Sonuclari sakla
        self._results[name] = {
            "model"     : model,
            "train_acc" : tr["accuracy"],
            "dev_acc"   : dv["accuracy"],
            "test_acc"  : te["accuracy"],
            "train_f1"  : tr["f1"],
            "dev_f1"    : dv["f1"],
            "test_f1"   : te["f1"],
            "n_steps"   : model.n_steps,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC METODLAR
    # ══════════════════════════════════════════════════════════════════════════

    def train_all_models(self):
        """
        5 modeli tanimlar ve sirayla egitir.

        Model 1 - Baseline   : NeuralNetwork     hidden=16, lr=0.01
        Model 2 - Derin      : DeepNeuralNetwork hidden=16, lr=0.01
        Model 3 - Genis      : NeuralNetwork     hidden=64, lr=0.01
        Model 4 - L2 Reg     : RegularizedNN     hidden=16, lambda=0.01
        Model 5 - Dropout    : DropoutNN         hidden=32, p=0.3
        """
        n_feat = self.X_train.shape[1]

        # -- Model 1: Baseline --
        m1 = NeuralNetwork(
            input_size=n_feat, hidden_size=16,
            learning_rate=0.01, n_steps=10_000, random_state=42
        )
        self._train_single("Model1_Baseline", m1)

        # -- Model 2: Derin (3 katman) --
        m2 = DeepNeuralNetwork(
            input_size=n_feat, hidden_size=16,
            learning_rate=0.01, n_steps=10_000, random_state=42
        )
        self._train_single("Model2_Derin", m2)

        # -- Model 3: Genis (hidden=64) --
        m3 = NeuralNetwork(
            input_size=n_feat, hidden_size=64,
            learning_rate=0.01, n_steps=10_000, random_state=42
        )
        self._train_single("Model3_Genis", m3)

        # -- Model 4: L2 Regularization --
        m4 = RegularizedNN(
            input_size=n_feat, hidden_size=16,
            learning_rate=0.01, n_steps=10_000,
            random_state=42, lambda_l2=0.01
        )
        self._train_single("Model4_L2Reg", m4)

        # -- Model 5: Dropout --
        m5 = DropoutNN(
            input_size=n_feat, hidden_size=32,
            learning_rate=0.01, n_steps=10_000,
            random_state=42, dropout_rate=0.3
        )
        self._train_single("Model5_Dropout", m5)

        print(f"\n[TRAINER] Tum modeller egitildi. Toplam: {len(self._results)}")

    # ──────────────────────────────────────────────────────────────────────────
    def compare_models(self):
        """
        Tum modellerin accuracy ve n_steps degerlerini tablo olarak yazdirir.
        """
        if not self._results:
            print("[UYARI] Once train_all_models() cagirin.")
            return

        header = f"{'Model':<22} {'Train Acc':>10} {'Dev Acc':>10} " \
                 f"{'Test Acc':>10} {'Train F1':>10} {'Test F1':>10} {'Steps':>8}"
        sep    = "-" * len(header)

        print(f"\n{'='*len(header)}")
        print("  MODEL KARSILASTIRMA TABLOSU")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)

        for name, r in self._results.items():
            print(f"{name:<22} "
                  f"{r['train_acc']:>10.4f} "
                  f"{r['dev_acc']:>10.4f} "
                  f"{r['test_acc']:>10.4f} "
                  f"{r['train_f1']:>10.4f} "
                  f"{r['test_f1']:>10.4f} "
                  f"{r['n_steps']:>8,}")
        print(sep)

    # ──────────────────────────────────────────────────────────────────────────
    def select_best_model(self, accuracy_threshold: float = 0.80):
        """
        Dev accuracy'si esigi gecen modeller arasinda en dusuk n_steps'e
        sahip olani secip yazdirir.

        Parametre
        ---------
        accuracy_threshold : minimum dev accuracy esigi (varsayilan 0.80)
        """
        if not self._results:
            print("[UYARI] Once train_all_models() cagirin.")
            return None

        # Esigi gecen adaylar
        candidates = {
            name: r for name, r in self._results.items()
            if r["dev_acc"] >= accuracy_threshold
        }

        print(f"\n[SECIM] Dev accuracy >= {accuracy_threshold} esigini gecen modeller:")
        if not candidates:
            print(f"  Hic model esigi gecemedi! "
                  f"(En iyi dev acc: "
                  f"{max(r['dev_acc'] for r in self._results.values()):.4f})")
            # Esik olmaksizin en iyi dev accuracy'ye sahip modeli sec
            best_name = max(self._results, key=lambda n: self._results[n]["dev_acc"])
        else:
            for name, r in candidates.items():
                print(f"  {name:<22} dev_acc={r['dev_acc']:.4f}  "
                      f"steps={r['n_steps']:,}")
            # En az adimla esigi gecen modeli sec
            best_name = min(candidates, key=lambda n: candidates[n]["n_steps"])

        best = self._results[best_name]
        print(f"\n  >> SECILEN MODEL: {best_name}")
        print(f"     Dev Acc  : {best['dev_acc']:.4f}")
        print(f"     Test Acc : {best['test_acc']:.4f}")
        print(f"     N Steps  : {best['n_steps']:,}")
        return best_name

    # ──────────────────────────────────────────────────────────────────────────
    def plot_comparison(self, save_path: str = "outputs/"):
        """
        Tum modellerin train ve dev loss egrilerini tek grafikte gosterir.
        Her model farkli renkte cizilir; dev loss kesikli cizgiyle gosterilir.

        Grafik 'model_comparison_loss.png' olarak kaydedilir.
        """
        if not self._results:
            print("[UYARI] Once train_all_models() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Model Karsilastirmasi — Train & Dev Loss",
                     fontsize=15, fontweight="bold")

        for idx, (name, r) in enumerate(self._results.items()):
            model  = r["model"]
            color  = self._COLORS[idx % len(self._COLORS)]
            label  = name.replace("_", " ")

            if model.train_losses:
                steps_tr, losses_tr = zip(*model.train_losses)
                axes[0].plot(steps_tr, losses_tr,
                             color=color, linewidth=1.8, label=label)

            if model.dev_losses:
                steps_dv, losses_dv = zip(*model.dev_losses)
                axes[1].plot(steps_dv, losses_dv,
                             color=color, linewidth=1.8,
                             linestyle="--", label=label)

        for ax, title in zip(axes, ["Train Loss", "Dev Loss"]):
            ax.set_xlabel("Adim (Step)", fontsize=11)
            ax.set_ylabel("BCE Loss", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.35)

        plt.tight_layout()
        out_file = os.path.join(save_path, "model_comparison_loss.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[PLOT] Karsilastirma grafigi kaydedildi -> {out_file}")

        # --- Ayrica accuracy bar chart ---
        self._plot_accuracy_bar(save_path)

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_accuracy_bar(self, save_path: str):
        """
        Her model icin Train / Dev / Test accuracy degerlerini
        gruplu bar chart olarak gosterir.
        """
        names      = list(self._results.keys())
        short_names = [n.replace("Model", "M").replace("_", "\n")
                       for n in names]
        train_accs = [self._results[n]["train_acc"] for n in names]
        dev_accs   = [self._results[n]["dev_acc"]   for n in names]
        test_accs  = [self._results[n]["test_acc"]  for n in names]

        x     = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        bars1 = ax.bar(x - width, train_accs, width,
                       label="Train", color="#4FC3F7", edgecolor="white")
        bars2 = ax.bar(x,          dev_accs,   width,
                       label="Dev",   color="#A5D6A7", edgecolor="white")
        bars3 = ax.bar(x + width,  test_accs,  width,
                       label="Test",  color="#EF9A9A", edgecolor="white")

        # Her cubuk uzerine deger yaz
        for bars in (bars1, bars2, bars3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0.80, color="crimson", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="Esik (0.80)")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_title("Model Bazinda Accuracy Karsilastirmasi",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.35)

        plt.tight_layout()
        out_file = os.path.join(save_path, "model_comparison_accuracy.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Accuracy bar chart kaydedildi -> {out_file}")

    # ──────────────────────────────────────────────────────────────────────────
    def save_results(self, save_path: str = "outputs/"):
        """
        Tum model sonuclarini CSV dosyasina kaydeder.

        Cikti : model_results.csv
        """
        if not self._results:
            print("[UYARI] Once train_all_models() cagirin.")
            return

        os.makedirs(save_path, exist_ok=True)
        csv_path = os.path.join(save_path, "model_results.csv")

        fieldnames = ["model", "train_acc", "dev_acc", "test_acc",
                      "train_f1", "dev_f1", "test_f1", "n_steps"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, r in self._results.items():
                writer.writerow({
                    "model"    : name,
                    "train_acc": round(r["train_acc"], 6),
                    "dev_acc"  : round(r["dev_acc"],   6),
                    "test_acc" : round(r["test_acc"],  6),
                    "train_f1" : round(r["train_f1"],  6),
                    "dev_f1"   : round(r["dev_f1"],    6),
                    "test_f1"  : round(r["test_f1"],   6),
                    "n_steps"  : r["n_steps"],
                })

        print(f"[CSV] Sonuclar kaydedildi -> {csv_path}")


# ==============================================================================
# MAIN BLOGU
# ==============================================================================
if __name__ == "__main__":

    # Yol ayarlari: bu dosya src/ icinde, proje koku bir ust dizin
    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")

    print("=" * 60)
    print("  YZM304 - TITANIC MODEL KARSILASTIRMA")
    print("=" * 60)

    # --- Trainer olustur ---
    trainer = ModelTrainer(
        processed_dir=PROCESSED_DIR,
        output_dir=OUTPUT_DIR,
    )

    # --- 1. Tum modelleri egit ---
    trainer.train_all_models()

    # --- 2. Karsilastirma tablosunu yazdir ---
    trainer.compare_models()

    # --- 3. En iyi modeli sec ---
    best = trainer.select_best_model(accuracy_threshold=0.80)

    # --- 4. Loss + Accuracy grafikleri ---
    trainer.plot_comparison(save_path=OUTPUT_DIR)

    # --- 5. CSV'ye kaydet ---
    trainer.save_results(save_path=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  TUM ISLEMLER TAMAMLANDI")
    print(f"  Ciktilar -> {OUTPUT_DIR}/")
    print("=" * 60)
