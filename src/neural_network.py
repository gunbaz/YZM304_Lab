"""
YZM304 - Derin Öğrenme | Lab 1
Titanic Binary Classification - Neural Network (Sadece NumPy)
Ankara Üniversitesi

Yazar : [Adınız Soyadınız]
Tarih : 2026-04-02

Mimari  : 22 → 16 (sigmoid) → 1 (sigmoid)
Loss    : Binary Cross Entropy
Optimizer: SGD
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Tek gizli katmanlı ikili sınıflandırma sinir ağı.

    Mimari: giriş(input_size) → gizli(hidden_size, sigmoid) → çıkış(1, sigmoid)

    Kullanım
    --------
    model = NeuralNetwork(input_size=22, hidden_size=16,
                          learning_rate=0.01, n_steps=10000)
    model.fit(X_train, y_train, X_dev, y_dev)
    model.evaluate(X_test, y_test, set_name="Test")
    model.plot_learning_curve(save_path="outputs/")
    model.save_weights("outputs/model_weights.npz")
    """

    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRUCTOR
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 16,
                 learning_rate: float = 0.01,
                 n_steps: int = 10_000,
                 random_state: int = 42):
        """
        Parametre
        ---------
        input_size    : giriş özellik sayısı (22)
        hidden_size   : gizli katman nöron sayısı (varsayılan 16)
        learning_rate : SGD öğrenme hızı       (varsayılan 0.01)
        n_steps       : toplam eğitim adımı    (varsayılan 10000)
        random_state  : tekrar üretilebilirlik için tohum
        """
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.lr            = learning_rate
        self.n_steps       = n_steps
        self.random_state  = random_state

        # Ağırlık matrisleri ve bias vektörleri
        self.W1 = None   # (hidden_size, input_size)
        self.b1 = None   # (hidden_size, 1)
        self.W2 = None   # (1, hidden_size)
        self.b2 = None   # (1, 1)

        # Forward pass önbelleği (backward için)
        self._cache = {}

        # Gradyanlar
        self._grads = {}

        # Eğitim sırasında kaydedilen kayıp listeleri
        self.train_losses = []
        self.dev_losses   = []

        # Ağırlıkları başlat
        self._initialize_weights()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC METODLAR
    # ══════════════════════════════════════════════════════════════════════════

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_dev: np.ndarray = None,
            y_dev: np.ndarray  = None) -> None:
        """
        Modeli eğitir.

        Her 1000 adımda train loss (ve varsa dev loss) ekrana yazdırılır.

        Parametre
        ---------
        X     : eğitim örnekleri  (n_samples, n_features)
        y     : eğitim etiketleri (n_samples,) veya (n_samples, 1)
        X_dev : doğrulama örnekleri (isteğe bağlı)
        y_dev : doğrulama etiketleri (isteğe bağlı)
        """
        # y'yi (n, 1) sütun vektörüne çevir
        y = y.reshape(-1, 1)
        if y_dev is not None:
            y_dev = y_dev.reshape(-1, 1)

        print("=" * 60)
        print("  EĞİTİM BAŞLIYOR")
        print(f"  Mimari  : {self.input_size} → {self.hidden_size} → 1")
        print(f"  Adım    : {self.n_steps:,}")
        print(f"  LR      : {self.lr}")
        print("=" * 60)

        self.train_losses = []
        self.dev_losses   = []

        for step in range(1, self.n_steps + 1):

            # ---------- Forward pass ----------
            y_pred = self._forward(X)

            # ---------- Loss hesapla ----------
            train_loss = self._compute_loss(y_pred, y)

            # ---------- Backward pass ----------
            self._backward(X, y)

            # ---------- Ağırlıkları güncelle ----------
            self._update_weights()

            # ---------- Her 1000 adımda kayıt ve ekran ----------
            if step % 1000 == 0 or step == 1:
                self.train_losses.append((step, float(train_loss)))

                log = f"  Adım {step:>6,} | Train Loss: {train_loss:.6f}"

                if X_dev is not None and y_dev is not None:
                    y_dev_pred = self._forward(X_dev)
                    dev_loss   = self._compute_loss(y_dev_pred, y_dev)
                    self.dev_losses.append((step, float(dev_loss)))
                    log += f" | Dev Loss: {dev_loss:.6f}"

                print(log)

        print("=" * 60)
        print("  EĞİTİM TAMAMLANDI")
        print("=" * 60)

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Sınıf tahmini döndürür (threshold = 0.5).

        Dönüş: (n_samples,) int dizisi — 0 veya 1
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int).flatten()

    # ──────────────────────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Ham sigmoid çıktısını (olasılık) döndürür.

        Dönüş: (n_samples,) float dizisi — [0, 1] aralığında
        """
        y_pred = self._forward(X)
        return y_pred.flatten()

    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 set_name: str = "Test") -> dict:
        """
        Accuracy, Precision, Recall ve F1 skorunu hesaplar ve yazdırır.

        Parametre
        ---------
        X        : özellik matrisi
        y        : gerçek etiketler
        set_name : ekran çıktısında kullanılacak isim

        Dönüş
        -----
        dict : {"accuracy", "precision", "recall", "f1"}
        """
        y_true = y.flatten()
        y_pred = self.predict(X)

        # Temel metrikler: TP, FP, FN, TN
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        accuracy  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        print(f"\n[{set_name} Değerlendirme]")
        print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        return {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1}

    # ──────────────────────────────────────────────────────────────────────────
    def plot_learning_curve(self, save_path: str = "outputs/") -> None:
        """
        Train ve Dev loss eğrilerini çizer ve PNG olarak kaydeder.

        Parametre
        ---------
        save_path : grafiklerin kaydedileceği klasör
        """
        if not self.train_losses:
            print("[UYARI] Henüz eğitim yapılmadı. Önce fit() çağrın.")
            return

        os.makedirs(save_path, exist_ok=True)

        steps_train, losses_train = zip(*self.train_losses)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps_train, losses_train,
                color="#4FC3F7", linewidth=2, label="Train Loss")

        if self.dev_losses:
            steps_dev, losses_dev = zip(*self.dev_losses)
            ax.plot(steps_dev, losses_dev,
                    color="#EF9A9A", linewidth=2,
                    linestyle="--", label="Dev Loss")

        ax.set_xlabel("Adım (Step)", fontsize=12)
        ax.set_ylabel("Binary Cross Entropy Loss", fontsize=12)
        ax.set_title("Öğrenme Eğrisi — Titanic Sinir Ağı",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        save_file = os.path.join(save_path, "learning_curve.png")
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n[PLOT] Ogrenme egrisi kaydedildi -> {save_file}")

    # ──────────────────────────────────────────────────────────────────────────
    def save_weights(self, path: str = "outputs/model_weights.npz") -> None:
        """
        Ağırlıkları ve bias'ları NumPy .npz formatında kaydeder.

        Parametre
        ---------
        path : kayıt dosyası yolu (.npz uzantılı olmalı)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 W1=self.W1,
                 b1=self.b1,
                 W2=self.W2,
                 b2=self.b2)
        print(f"[SAVE] Agirliklar kaydedildi -> {path}")

    # ──────────────────────────────────────────────────────────────────────────
    def load_weights(self, path: str) -> None:
        """
        Daha önce kaydedilmiş ağırlıkları yükler.

        Parametre
        ---------
        path : .npz dosyasının yolu
        """
        data    = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        print(f"[LOAD] Agirliklar yuklendi <- {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE METODLAR
    # ══════════════════════════════════════════════════════════════════════════

    def _initialize_weights(self) -> None:
        """
        Xavier (Glorot) başlatma yöntemi ile ağırlıkları başlatır.

        Xavier formülü: std = sqrt(2 / (fan_in + fan_out))
        Bu yöntem sigmoid aktivasyonu için vanishing gradient'i azaltır.
        """
        np.random.seed(self.random_state)

        # Gizli katman: W1 → (hidden_size, input_size)
        xavier_std_1 = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * xavier_std_1
        self.b1 = np.zeros((self.hidden_size, 1))

        # Çıkış katmanı: W2 → (1, hidden_size)
        xavier_std_2 = np.sqrt(2.0 / (self.hidden_size + 1))
        self.W2 = np.random.randn(1, self.hidden_size) * xavier_std_2
        self.b2 = np.zeros((1, 1))

    # ──────────────────────────────────────────────────────────────────────────
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sayısal olarak kararlı sigmoid aktivasyon fonksiyonu.

        σ(z) = 1 / (1 + e^(-z))

        Taşma (overflow) önlemek için z'yi [-500, 500] arasında klipliyor.
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    # ──────────────────────────────────────────────────────────────────────────
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        İleri yayılım (forward pass).

        X : (n_samples, n_features)
        Dönüş: y_pred → (n_samples, 1)

        İşlem akışı:
            Z1 = W1 · X^T + b1        → (hidden_size, n_samples)
            A1 = sigmoid(Z1)           → (hidden_size, n_samples)
            Z2 = W2 · A1 + b2          → (1, n_samples)
            A2 = sigmoid(Z2)           → (1, n_samples)
        """
        X_T = X.T   # (n_features, n_samples)

        # -- Gizli katman --
        Z1 = self.W1 @ X_T + self.b1   # (hidden_size, n_samples)
        A1 = self._sigmoid(Z1)          # (hidden_size, n_samples)

        # -- Çıkış katmanı --
        Z2 = self.W2 @ A1 + self.b2    # (1, n_samples)
        A2 = self._sigmoid(Z2)          # (1, n_samples)

        # Backward için önbelleğe al
        self._cache = {
            "X" : X,
            "Z1": Z1, "A1": A1,
            "Z2": Z2, "A2": A2,
        }

        return A2.T   # (n_samples, 1)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_loss(self,
                      y_pred: np.ndarray,
                      y_true: np.ndarray) -> float:
        """
        Binary Cross Entropy Loss hesaplar.

        L = -1/N · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

        epsilon = 1e-15 ile log(0) hatasından kaçınılır.

        Parametre
        ---------
        y_pred : (n_samples, 1) — sigmoid çıktısı
        y_true : (n_samples, 1) — gerçek etiketler

        Dönüş: skalar kayıp değeri
        """
        eps = 1e-15
        n   = y_true.shape[0]

        # log sıfırdan geçmemesi için klipleme
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        return float(loss)

    # ──────────────────────────────────────────────────────────────────────────
    def _backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Geri yayılım (backpropagation).

        Zincir kuralıyla tüm ağırlıklar için gradyanı hesaplar
        ve self._grads sözlüğüne kaydeder.

        Türevler (sigmoid için: σ'(z) = σ(z)·(1−σ(z))):
            dZ2 = A2 - y^T                             (çıkış katmanı delta)
            dW2 = (1/N) · dZ2 · A1^T
            db2 = (1/N) · sum(dZ2, axis=1, keepdim)
            dA1 = W2^T · dZ2
            dZ1 = dA1 · A1·(1−A1)                     (sigmoid türevi)
            dW1 = (1/N) · dZ1 · X
            db1 = (1/N) · sum(dZ1, axis=1, keepdim)
        """
        A1 = self._cache["A1"]           # (hidden_size, n_samples)
        A2 = self._cache["A2"]           # (1, n_samples)
        n  = X.shape[0]

        # y'yi (1, n_samples) boyutuna getir
        y_T = y.T   # (1, n_samples)

        # -- Çıkış katmanı gradyanları --
        dZ2 = A2 - y_T                              # (1, n_samples)
        dW2 = (1.0 / n) * (dZ2 @ A1.T)             # (1, hidden_size)
        db2 = (1.0 / n) * np.sum(dZ2, axis=1,
                                  keepdims=True)    # (1, 1)

        # -- Gizli katman gradyanları --
        dA1 = self.W2.T @ dZ2                       # (hidden_size, n_samples)
        dZ1 = dA1 * A1 * (1.0 - A1)                # sigmoid türevi
        dW1 = (1.0 / n) * (dZ1 @ X)                # (hidden_size, input_size)
        db1 = (1.0 / n) * np.sum(dZ1, axis=1,
                                  keepdims=True)    # (hidden_size, 1)

        self._grads = {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _update_weights(self) -> None:
        """
        Stochastic Gradient Descent (SGD) ile ağırlıkları günceller.

        Kural:  θ ← θ - lr · ∂L/∂θ
        """
        self.W1 -= self.lr * self._grads["dW1"]
        self.b1 -= self.lr * self._grads["db1"]
        self.W2 -= self.lr * self._grads["dW2"]
        self.b2 -= self.lr * self._grads["db2"]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BLOĞU
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Yol ayarları ──────────────────────────────────────────────────────────
    BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Veriyi Yükle ───────────────────────────────────────────────────────
    print("Veri yükleniyor...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"), allow_pickle=True)
    X_dev   = np.load(os.path.join(PROCESSED_DIR, "X_dev.npy"),   allow_pickle=True)
    y_dev   = np.load(os.path.join(PROCESSED_DIR, "y_dev.npy"),   allow_pickle=True)
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"),  allow_pickle=True)
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"),  allow_pickle=True)

    # object array ise float'a çevir
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    X_dev   = X_dev.astype(np.float64)
    y_dev   = y_dev.astype(np.float64)
    X_test  = X_test.astype(np.float64)
    y_test  = y_test.astype(np.float64)

    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_dev  : {X_dev.shape}   | y_dev  : {y_dev.shape}")
    print(f"  X_test : {X_test.shape}  | y_test : {y_test.shape}")

    # ── 2. Modeli Oluştur ─────────────────────────────────────────────────────
    model = NeuralNetwork(
        input_size    = X_train.shape[1],  # 22
        hidden_size   = 16,
        learning_rate = 0.01,
        n_steps       = 10_000,
        random_state  = 42,
    )

    # ── 3. Eğit ───────────────────────────────────────────────────────────────
    model.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)

    # ── 4. Değerlendirme ──────────────────────────────────────────────────────
    model.evaluate(X_train, y_train, set_name="Train")
    model.evaluate(X_dev,   y_dev,   set_name="Dev  ")
    model.evaluate(X_test,  y_test,  set_name="Test ")

    # ── 5. Öğrenme Eğrisi ─────────────────────────────────────────────────────
    model.plot_learning_curve(save_path=OUTPUT_DIR)

    # ── 6. Ağırlıkları Kaydet ─────────────────────────────────────────────────
    model.save_weights(os.path.join(OUTPUT_DIR, "model_weights.npz"))

    print("\n" + "=" * 60)
    print("  TÜM İŞLEMLER TAMAMLANDI")
    print(f"  Ciktilar -> {OUTPUT_DIR}/")
    print("=" * 60)
