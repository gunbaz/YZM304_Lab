"""
YZM304 - Derin Öğrenme | Lab 1
Titanic Binary Classification - Data Preprocessing Pipeline
Ankara Üniversitesi

Yazar : [Adınız Soyadınız]
Tarih : 2026-04-02
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ──────────────────────────────────────────────────────────────────────────────
# Görsel tema ayarları
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = {"survived": "#4CAF50", "died": "#F44336"}
FIG_DPI = 150


class TitanicDataProcessor:
    """
    Titanic veri seti için uçtan uca ön işleme pipeline'ı.

    Kullanım örneği
    ---------------
    processor = TitanicDataProcessor(data_path="data/raw/titanic.csv",
                                     output_dir="outputs/eda",
                                     processed_dir="data/processed")
    processor.load_data()
    processor.run_eda()
    processor.preprocess(scaler_type="standard")
    processor.split_data(train=0.70, dev=0.15, test=0.15)
    processor.save_processed_data()
    X_train, y_train, X_dev, y_dev, X_test, y_test = processor.get_splits()
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Constructor
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self,
                 data_path: str = "data/raw/titanic.csv",
                 output_dir: str = "outputs/eda",
                 processed_dir: str = "data/processed"):
        """
        Parameters
        ----------
        data_path     : ham CSV dosyasının yolu
        output_dir    : EDA grafiklerinin kaydedileceği klasör
        processed_dir : işlenmiş dosyaların kaydedileceği klasör
        """
        self.data_path     = data_path
        self.output_dir    = output_dir
        self.processed_dir = processed_dir

        self.df            = None   # ham DataFrame
        self._df_processed = None   # ön işlenmiş DataFrame
        self._scaler       = None   # fit edilmiş scaler nesnesi
        self._scaled_cols  = []     # ölçeklenen sütun isimleri

        # split sonuçları
        self._X_train = self._X_dev = self._X_test = None
        self._y_train = self._y_dev = self._y_test = None

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC METODLAR
    # ══════════════════════════════════════════════════════════════════════════

    def load_data(self) -> pd.DataFrame:
        """CSV dosyasını yükler ve temel bilgileri ekranda gösterir."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"\n[!] '{self.data_path}' bulunamadı.\n"
                "    Lütfen Kaggle'dan titanic.csv dosyasını indirin:\n"
                "    https://www.kaggle.com/competitions/titanic/data\n"
                "    ve 'data/raw/titanic.csv' yoluna yerleştirin."
            )

        self.df = pd.read_csv(self.data_path)
        print("=" * 60)
        print("  VERİ YÜKLENDİ")
        print("=" * 60)
        print(f"  Satır × Sütun : {self.df.shape}")
        print(f"  Sütunlar      : {list(self.df.columns)}")
        print(f"\n  Sınıf Dağılımı:")
        vc = self.df["Survived"].value_counts()
        print(f"    Hayatta kalan (1) : {vc.get(1, 0)} "
              f"({vc.get(1, 0)/len(self.df)*100:.1f}%)")
        print(f"    Hayatta kalmayan (0): {vc.get(0, 0)} "
              f"({vc.get(0, 0)/len(self.df)*100:.1f}%)")
        print("=" * 60)
        return self.df

    # ──────────────────────────────────────────────────────────────────────────
    def run_eda(self) -> None:
        """EDA grafiklerini oluşturur ve outputs/eda/ klasörüne kaydeder."""
        self._check_loaded("run_eda")
        print("\n[EDA] Grafikler oluşturuluyor...")

        self._plot_survival_distribution()
        self._plot_missing_values()
        self._plot_feature_distributions()
        self._plot_correlation_matrix()

        print(f"[EDA] Tüm grafikler '{self.output_dir}/' klasörüne kaydedildi.\n")

    # ──────────────────────────────────────────────────────────────────────────
    def preprocess(self, scaler_type: str = "standard") -> pd.DataFrame:
        """
        Tam ön işleme pipeline'ını çalıştırır.

        Parameters
        ----------
        scaler_type : "standard" → StandardScaler | "minmax" → MinMaxScaler
        """
        self._check_loaded("preprocess")
        self._df_processed = self.df.copy()

        print("\n[PREPROCESS] Pipeline başlıyor...")
        self._handle_missing_values()
        self._feature_engineering()
        self._encode_categoricals()
        self._drop_irrelevant()
        self._scale_features(scaler_type)

        print(f"[PREPROCESS] Tamamlandı. "
              f"Son shape: {self._df_processed.shape}")
        print(f"  Sütunlar: {list(self._df_processed.columns)}\n")
        return self._df_processed

    # ──────────────────────────────────────────────────────────────────────────
    def split_data(self,
                   train: float = 0.70,
                   dev: float   = 0.15,
                   test: float  = 0.15) -> None:
        """
        Stratified train / dev / test bölmesi yapar.

        Parameters
        ----------
        train : eğitim oranı  (varsayılan 0.70)
        dev   : doğrulama oranı (varsayılan 0.15)
        test  : test oranı    (varsayılan 0.15)
        """
        assert abs(train + dev + test - 1.0) < 1e-6, \
            "train + dev + test toplamı 1.0 olmalı!"
        self._check_processed("split_data")

        X = self._df_processed.drop(columns=["Survived"]).values
        y = self._df_processed["Survived"].values

        # İlk olarak train / (dev+test) split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1.0 - train),
            stratify=y,
            random_state=42
        )

        # Sonra (dev+test) → dev / test
        relative_test = test / (dev + test)
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test,
            stratify=y_temp,
            random_state=42
        )

        self._X_train, self._y_train = X_train, y_train
        self._X_dev,   self._y_dev   = X_dev,   y_dev
        self._X_test,  self._y_test  = X_test,  y_test

        print("[SPLIT] Veri bölmeleri:")
        print(f"  Train : {X_train.shape[0]} örnek")
        print(f"  Dev   : {X_dev.shape[0]} örnek")
        print(f"  Test  : {X_test.shape[0]} örnek\n")

        self._check_class_balance()

    # ──────────────────────────────────────────────────────────────────────────
    def save_processed_data(self) -> None:
        """İşlenmiş veriyi ve bölmeleri data/processed/ klasörüne kaydeder."""
        self._check_splits("save_processed_data")

        feature_cols = [c for c in self._df_processed.columns if c != "Survived"]

        # Tam işlenmiş DataFrame
        self._df_processed.to_csv(
            os.path.join(self.processed_dir, "titanic_processed.csv"),
            index=False
        )

        # NumPy dizileri
        for name, arr in [
            ("X_train", self._X_train), ("y_train", self._y_train),
            ("X_dev",   self._X_dev),   ("y_dev",   self._y_dev),
            ("X_test",  self._X_test),  ("y_test",  self._y_test),
        ]:
            np.save(os.path.join(self.processed_dir, f"{name}.npy"), arr)

        # Özellik isimleri
        pd.Series(feature_cols).to_csv(
            os.path.join(self.processed_dir, "feature_names.csv"),
            index=False, header=False
        )

        print(f"[SAVE] İşlenmiş veriler '{self.processed_dir}/' klasörüne kaydedildi.")
        print(f"  • titanic_processed.csv")
        print(f"  • X_train.npy / y_train.npy")
        print(f"  • X_dev.npy   / y_dev.npy")
        print(f"  • X_test.npy  / y_test.npy")
        print(f"  • feature_names.csv\n")

    # ──────────────────────────────────────────────────────────────────────────
    def get_splits(self):
        """
        Bölünmüş numpy dizilerini döndürür.

        Returns
        -------
        X_train, y_train, X_dev, y_dev, X_test, y_test : np.ndarray
        """
        self._check_splits("get_splits")
        return (self._X_train, self._y_train,
                self._X_dev,   self._y_dev,
                self._X_test,  self._y_test)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE METODLAR — ön işleme adımları
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_missing_values(self) -> None:
        """
        Eksik değerleri doldurur:
        - Age      → Pclass + Sex grubuna göre medyan
        - Embarked → mod (en sık değer)
        - Cabin    → Has_Cabin ikili özelliğine dönüştür → Cabin sütunu düşürülür
        """
        df = self._df_processed

        # Cabin → Has_Cabin
        df["Has_Cabin"] = df["Cabin"].notna().astype(int)

        # Age → grup medyanı
        age_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
        df["Age"] = df["Age"].fillna(age_medians)
        # Hâlâ eksik kalan (grup medyanı da NaN ise) → genel medyanla doldur
        df["Age"] = df["Age"].fillna(df["Age"].median())

        # Embarked → mod
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        self._df_processed = df
        print("  [1/5] Eksik değerler işlendi.")

    # ──────────────────────────────────────────────────────────────────────────
    def _feature_engineering(self) -> None:
        """
        Yeni özellikler türetir:
        - FamilySize  = SibSp + Parch + 1
        - IsAlone     = 1 iff FamilySize == 1
        - Title       = isimden regex ile çekilen unvan; nadir → 'Rare'
        - AgeBin      = 5 eşit aralıklı yaş gruba
        - FareBin     = 4 çeyreklik ücret grubuna (quartile-based)
        - Age_Pclass  = Age × Pclass (etkileşim özelliği)
        """
        df = self._df_processed

        # FamilySize ve IsAlone
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

        # Title — regex: "Soyadı, Unvan. Ad" formatı
        df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.")
        df["Title"] = df["Title"].str.strip()

        # Nadir unvanları 'Rare' olarak etiketle
        common_titles = {"Mr", "Miss", "Mrs", "Master"}
        df["Title"] = df["Title"].apply(
            lambda t: t if t in common_titles else "Rare"
        )

        # AgeBin — 5 eşit aralık
        df["AgeBin"] = pd.cut(
            df["Age"], bins=5,
            labels=["VeryYoung", "Young", "Middle", "Senior", "Old"]
        )

        # FareBin — 4 çeyreklik (quantile)
        df["FareBin"] = pd.qcut(
            df["Fare"], q=4,
            labels=["Low", "Medium", "High", "VeryHigh"],
            duplicates="drop"
        )

        # Etkileşim özelliği
        df["Age_Pclass"] = df["Age"] * df["Pclass"]

        self._df_processed = df
        print("  [2/5] Özellik mühendisliği tamamlandı.")

    # ──────────────────────────────────────────────────────────────────────────
    def _encode_categoricals(self) -> None:
        """
        Kategorik sütunları kodlar:
        - Sex      → binary  (male=1, female=0)
        - Embarked → one-hot (drop_first=True)
        - Title    → one-hot (drop_first=True)
        - AgeBin   → one-hot (drop_first=True)
        - FareBin  → one-hot (drop_first=True)
        """
        df = self._df_processed

        # Sex → binary
        df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

        # One-hot encoding
        for col in ["Embarked", "Title", "AgeBin", "FareBin"]:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

        self._df_processed = df
        print("  [3/5] Kategorik kodlama tamamlandı.")

    # ──────────────────────────────────────────────────────────────────────────
    def _drop_irrelevant(self) -> None:
        """Modelle ilgisiz / bilgi sızdıran sütunları düşürür."""
        drop_cols = ["PassengerId", "Name", "Ticket", "Cabin",
                     "Sex", "Embarked", "Title"]
        # Yalnızca gerçekten mevcut olanları düşür
        existing = [c for c in drop_cols if c in self._df_processed.columns]
        self._df_processed.drop(columns=existing, inplace=True)
        print(f"  [4/5] Gereksiz sütunlar düşürüldü: {existing}")

    # ──────────────────────────────────────────────────────────────────────────
    def _scale_features(self, scaler_type: str) -> None:
        """
        Sayısal sürekli sütunlara ölçekleme uygular.
        Binary / dummy sütunlar ölçeklenmez.

        Parameters
        ----------
        scaler_type : "standard" | "minmax"
        """
        df = self._df_processed

        # Binary / dummy sütunları tespiti: sadece {0, 1} değeri alanlar
        binary_cols = [
            c for c in df.columns
            if c != "Survived" and df[c].dropna().isin([0, 1]).all()
        ]

        # Ölçeklenecek sütunlar
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in binary_cols and c != "Survived"
        ]

        if scaler_type == "standard":
            self._scaler = StandardScaler()
        elif scaler_type == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Geçersiz scaler_type: '{scaler_type}'. "
                             "'standard' veya 'minmax' kullanın.")

        df[num_cols] = self._scaler.fit_transform(df[num_cols])
        self._scaled_cols      = num_cols
        self._df_processed     = df

        print(f"  [5/5] Ölçekleme ({scaler_type}) uygulandı: {num_cols}")

    # ──────────────────────────────────────────────────────────────────────────
    def _check_class_balance(self) -> None:
        """Her bölmede sınıf oranlarını ekrana yazdırır."""
        splits = {
            "Train": (self._X_train, self._y_train),
            "Dev"  : (self._X_dev,   self._y_dev),
            "Test" : (self._X_test,  self._y_test),
        }
        print("[CLASS BALANCE]")
        for name, (_, y) in splits.items():
            n_total   = len(y)
            n_survived = int(y.sum())
            n_died     = n_total - n_survived
            print(f"  {name:6s} → Hayatta kalan: {n_survived} "
                  f"({n_survived/n_total*100:.1f}%) | "
                  f"Hayatta kalmayan: {n_died} "
                  f"({n_died/n_total*100:.1f}%)")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE METODLAR — EDA grafikleri
    # ══════════════════════════════════════════════════════════════════════════

    def _plot_survival_distribution(self) -> None:
        """
        eda_survival_distribution.png
        3 panel: genel dağılım | cinsiyete göre | Pclass'a göre
        """
        df = self.df
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Hayatta Kalma Dağılımı", fontsize=16, fontweight="bold", y=1.01)

        # Panel 1 — genel
        counts = df["Survived"].value_counts().sort_index()
        axes[0].bar(["Hayatta Kalmadı (0)", "Hayatta Kaldı (1)"],
                    counts.values,
                    color=[PALETTE["died"], PALETTE["survived"]],
                    edgecolor="white", linewidth=1.2)
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
        axes[0].set_title("Genel Dağılım")
        axes[0].set_ylabel("Yolcu Sayısı")

        # Panel 2 — cinsiyete göre
        sex_surv = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
        sex_surv.index = ["Kadın", "Erkek"]
        sex_surv.columns = ["Hayatta Kalmadı", "Hayatta Kaldı"]
        sex_surv.plot(kind="bar", ax=axes[1],
                      color=[PALETTE["died"], PALETTE["survived"]],
                      edgecolor="white", linewidth=1.2, rot=0)
        axes[1].set_title("Cinsiyete Göre Hayatta Kalma")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Yolcu Sayısı")
        axes[1].legend(loc="upper right", fontsize=9)

        # Panel 3 — Pclass'a göre
        cls_surv = df.groupby(["Pclass", "Survived"]).size().unstack(fill_value=0)
        cls_surv.index = [f"{i}. Sınıf" for i in cls_surv.index]
        cls_surv.columns = ["Hayatta Kalmadı", "Hayatta Kaldı"]
        cls_surv.plot(kind="bar", ax=axes[2],
                      color=[PALETTE["died"], PALETTE["survived"]],
                      edgecolor="white", linewidth=1.2, rot=0)
        axes[2].set_title("Yolcu Sınıfına Göre Hayatta Kalma")
        axes[2].set_xlabel("")
        axes[2].set_ylabel("Yolcu Sayısı")
        axes[2].legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "eda_survival_distribution.png")
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi → {path}")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_missing_values(self) -> None:
        """
        eda_missing_values.png
        Eksik değer yüzdelerini bar chart olarak gösterir.
        """
        missing = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        missing = missing[missing > 0]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(missing.index, missing.values,
                       color=sns.color_palette("Reds_r", len(missing)),
                       edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, missing.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("Eksik Değer Yüzdesi (%)", fontsize=12)
        ax.set_title("Sütunlara Göre Eksik Değer Oranları", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.axvline(x=50, color="crimson", linestyle="--", linewidth=1, alpha=0.7,
                   label="%50 Eşiği")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, "eda_missing_values.png")
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi → {path}")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_feature_distributions(self) -> None:
        """
        eda_feature_distributions.png
        Age, Fare, SibSp, Parch için histogram + boxplot (2×4 grid)
        """
        features = ["Age", "Fare", "SibSp", "Parch"]
        fig = plt.figure(figsize=(18, 8))
        fig.suptitle("Özellik Dağılımları", fontsize=16, fontweight="bold")
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        colors = sns.color_palette("husl", len(features))

        for col_idx, (feat, color) in enumerate(zip(features, colors)):
            series = self.df[feat].dropna()

            # Üst satır — histogram + KDE
            ax_hist = fig.add_subplot(gs[0, col_idx])
            ax_hist.hist(series, bins=30, color=color, edgecolor="white",
                         linewidth=0.5, alpha=0.85, density=True)
            series.plot.kde(ax=ax_hist, color="white", linewidth=1.5)
            ax_hist.set_title(f"{feat} — Histogram", fontsize=10)
            ax_hist.set_xlabel("")
            ax_hist.set_ylabel("Yoğunluk" if col_idx == 0 else "")

            # Alt satır — boxplot (survived göre renklendirilmiş)
            ax_box = fig.add_subplot(gs[1, col_idx])
            groups = [
                self.df.loc[self.df["Survived"] == 0, feat].dropna(),
                self.df.loc[self.df["Survived"] == 1, feat].dropna(),
            ]
            bp = ax_box.boxplot(groups, patch_artist=True,
                                medianprops={"color": "white", "linewidth": 2})
            for patch, c in zip(bp["boxes"],
                                [PALETTE["died"], PALETTE["survived"]]):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)
            ax_box.set_xticklabels(["Kalmadı", "Kaldı"], fontsize=9)
            ax_box.set_title(f"{feat} — Boxplot", fontsize=10)
            if col_idx == 0:
                ax_box.set_ylabel("Değer")

        path = os.path.join(self.output_dir, "eda_feature_distributions.png")
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi → {path}")

    # ──────────────────────────────────────────────────────────────────────────
    def _plot_correlation_matrix(self) -> None:
        """
        eda_correlation_matrix.png
        Sayısal sütunların korelasyon ısı haritası.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        corr   = num_df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="gray",
            ax=ax, annot_kws={"size": 9}
        )
        ax.set_title("Korelasyon Matrisi (Sayısal Özellikler)",
                     fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "eda_correlation_matrix.png")
        plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
        print(f"  Kaydedildi → {path}")

    # ══════════════════════════════════════════════════════════════════════════
    # YARDIMCI (GUARD) METODLAR
    # ══════════════════════════════════════════════════════════════════════════

    def _check_loaded(self, caller: str) -> None:
        if self.df is None:
            raise RuntimeError(
                f"[{caller}] Önce load_data() çağrılmalı."
            )

    def _check_processed(self, caller: str) -> None:
        if self._df_processed is None:
            raise RuntimeError(
                f"[{caller}] Önce preprocess() çağrılmalı."
            )

    def _check_splits(self, caller: str) -> None:
        if self._X_train is None:
            raise RuntimeError(
                f"[{caller}] Önce split_data() çağrılmalı."
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BLOĞU — Pipeline sırayla çalıştırılır
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Yol ayarları ──────────────────────────────────────────────────────────
    # Bu script'in bulunduğu klasörden bir üst dizin (titanic_project/) kökü
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_PATH     = os.path.join(BASE_DIR, "data", "raw", "titanic.csv")
    OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "eda")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    # ── Processor oluştur ─────────────────────────────────────────────────────
    processor = TitanicDataProcessor(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        processed_dir=PROCESSED_DIR
    )

    # ── 1. Veriyi Yükle ───────────────────────────────────────────────────────
    processor.load_data()

    # ── 2. Keşifsel Veri Analizi ──────────────────────────────────────────────
    processor.run_eda()

    # ── 3. Ön İşleme ──────────────────────────────────────────────────────────
    processor.preprocess(scaler_type="standard")

    # ── 4. Veriyi Böl ─────────────────────────────────────────────────────────
    processor.split_data(train=0.70, dev=0.15, test=0.15)

    # ── 5. Kaydet ─────────────────────────────────────────────────────────────
    processor.save_processed_data()

    # ── 6. Split'leri Al ve Doğrula ───────────────────────────────────────────
    X_train, y_train, X_dev, y_dev, X_test, y_test = processor.get_splits()

    print("=" * 60)
    print("  PIPELINE TAMAMLANDI")
    print("=" * 60)
    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_dev   : {X_dev.shape}    |  y_dev   : {y_dev.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")
    print("=" * 60)
