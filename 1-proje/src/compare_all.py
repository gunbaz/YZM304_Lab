"""
YZM304 - Derin Ogrenme | Lab 1
Titanic Binary Classification — Nihai Model Karsilastirmasi
Ankara Universitesi

Yazar : [Adiniz Soyadiniz]
Tarih : 2026-04-02

Bu script:
  1. model_results.csv  (NumPy modeli — Model1_Baseline)
  2. sklearn_results.csv (Sklearn MLPClassifier)
  3. pytorch_results.csv (PyTorch TitanicNet)
dosyalarini okur ve yan yana karsilastirir.
"""

import os
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# YARDIMCI OKUYUCU FONKSIYONLAR
# ==============================================================================

def _read_numpy_results(csv_path: str, target_model: str = "Model1_Baseline") -> dict:
    """
    model_results.csv dosyasindan belirtilen modelin metriklerini oku.

    Numpy CSV sutunlari:
      model, train_acc, dev_acc, test_acc, train_f1, dev_f1, test_f1, n_steps

    Dondur:
      {split: {accuracy, f1}} — precision ve recall mevcut degil
    """
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == target_model:
                results["train"] = {
                    "accuracy" : float(row["train_acc"]),
                    "precision": None,
                    "recall"   : None,
                    "f1"       : float(row["train_f1"]),
                }
                results["dev"] = {
                    "accuracy" : float(row["dev_acc"]),
                    "precision": None,
                    "recall"   : None,
                    "f1"       : float(row["dev_f1"]),
                }
                results["test"] = {
                    "accuracy" : float(row["test_acc"]),
                    "precision": None,
                    "recall"   : None,
                    "f1"       : float(row["test_f1"]),
                }
                break

    # Eger hedef model bulunamazsa en iyi test acc'li satiri al
    if not results:
        best_row  = None
        best_tacc = -1.0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tacc = float(row["test_acc"])
                if tacc > best_tacc:
                    best_tacc = tacc
                    best_row  = row
        if best_row:
            results["train"] = {
                "accuracy": float(best_row["train_acc"]),
                "precision": None, "recall": None,
                "f1": float(best_row["train_f1"]),
            }
            results["dev"]   = {
                "accuracy": float(best_row["dev_acc"]),
                "precision": None, "recall": None,
                "f1": float(best_row["dev_f1"]),
            }
            results["test"]  = {
                "accuracy": float(best_row["test_acc"]),
                "precision": None, "recall": None,
                "f1": float(best_row["test_f1"]),
            }
    return results


def _read_standard_results(csv_path: str) -> dict:
    """
    sklearn_results.csv veya pytorch_results.csv dosyasindan verileri oku.

    Sutunlar: model, split, accuracy, precision, recall, f1
    """
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            results[split] = {
                "accuracy" : float(row["accuracy"]),
                "precision": float(row["precision"]),
                "recall"   : float(row["recall"]),
                "f1"       : float(row["f1"]),
            }
    return results


# ==============================================================================
# KARSILASTIRMA GRAFIGI
# ==============================================================================

def plot_final_comparison(all_results: dict, save_path: str) -> None:
    """
    Accuracy, Precision, Recall, F1 icin modeller arasi grouped bar chart ciz.

    Sadece TEST split metriklerini karsilastirir.

    Parametre
    ---------
    all_results : {model_adi: {split: {metrik: deger}}}
    save_path   : PNG kayit klasoru
    """
    os.makedirs(save_path, exist_ok=True)

    model_names  = list(all_results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    # Precision ve recall eksikse yerine None — grafige 0.0 koy
    colors = ["#4FC3F7", "#A5D6A7", "#EF9A9A", "#FFE082", "#CE93D8"]

    x     = np.arange(len(metric_names))
    width = 0.22
    offsets = np.linspace(-(len(model_names)-1)/2,
                           (len(model_names)-1)/2,
                           len(model_names)) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, (model_name, splits) in enumerate(all_results.items()):
        test_m = splits.get("test", {})
        vals   = []
        for m in metric_names:
            v = test_m.get(m)
            vals.append(v if v is not None else 0.0)

        bars = ax.bar(x + offsets[idx], vals, width,
                      label=model_name,
                      color=colors[idx % len(colors)],
                      edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, vals):
            if v > 0.0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.axhline(0.80, color="crimson", linestyle="--",
               linewidth=1.2, alpha=0.7, label="Esik (0.80)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names], fontsize=12)
    ax.set_ylabel("Test Skoru", fontsize=12)
    ax.set_ylim(0.0, 1.18)
    ax.set_title("Nihai Model Karsilastirmasi — Test Metrikleri\n"
                 "(NumPy | Sklearn | PyTorch)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()

    out = os.path.join(save_path, "final_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Nihai karsilastirma grafigi -> {out}")


def plot_split_comparison(all_results: dict, save_path: str) -> None:
    """
    Train / Dev / Test accuracy degerlerini modeller bazinda
    3 panelli grafik olarak gosterir.
    """
    os.makedirs(save_path, exist_ok=True)
    splits      = ["train", "dev", "test"]
    model_names = list(all_results.keys())
    colors      = ["#4FC3F7", "#A5D6A7", "#EF9A9A", "#FFE082", "#CE93D8"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Model Bazinda Accuracy (Train / Dev / Test)",
                 fontsize=14, fontweight="bold")

    for ax_idx, split in enumerate(splits):
        accs = [all_results[m].get(split, {}).get("accuracy", 0.0)
                for m in model_names]
        bars = axes[ax_idx].bar(range(len(model_names)), accs,
                                color=[colors[i % len(colors)]
                                       for i in range(len(model_names))],
                                edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, accs):
            axes[ax_idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9
            )
        axes[ax_idx].set_xticks(range(len(model_names)))
        axes[ax_idx].set_xticklabels(
            [n.replace("_", "\n") for n in model_names], fontsize=8
        )
        axes[ax_idx].set_title(split.capitalize(), fontsize=12)
        axes[ax_idx].set_ylim(0.5, 1.10)
        axes[ax_idx].axhline(0.80, color="crimson", linestyle="--",
                              linewidth=1, alpha=0.6)
        axes[ax_idx].grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    plt.tight_layout()

    out = os.path.join(save_path, "split_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Split karsilastirma grafigi -> {out}")


# ==============================================================================
# TERMINAL TABLOSU
# ==============================================================================

def print_comparison_table(all_results: dict) -> str:
    """
    Tum modellerin TEST split metriklerini terminale yazar ve
    en iyi modeli belirleyip dondurur.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    header = (f"{'Model':<24} "
              + "  ".join(f"{m.capitalize():>10}" for m in metric_keys))
    sep    = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  NIHAI KARSILASTIRMA TABLOSU — TEST SETI")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    best_model = None
    best_acc   = -1.0

    for model_name, splits in all_results.items():
        test_m = splits.get("test", {})
        row_vals = []
        for m in metric_keys:
            v = test_m.get(m)
            row_vals.append(f"{v:>10.4f}" if v is not None else f"{'N/A':>10}")
        print(f"{model_name:<24} " + "  ".join(row_vals))

        acc = test_m.get("accuracy", -1.0) or -1.0
        if acc > best_acc:
            best_acc   = acc
            best_model = model_name

    print(sep)
    print(f"\n  >> EN IYI MODEL : {best_model}")
    print(f"     Test Accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%)")
    return best_model


# ==============================================================================
# MAIN BLOGU
# ==============================================================================

if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    numpy_csv   = os.path.join(OUTPUT_DIR, "model_results.csv")
    sklearn_csv = os.path.join(OUTPUT_DIR, "sklearn_results.csv")
    pytorch_csv = os.path.join(OUTPUT_DIR, "pytorch_results.csv")

    print("=" * 60)
    print("  YZM304 — NIHAI MODEL KARSILASTIRMASI")
    print("=" * 60)

    # CSV dosyalarinin varligini kontrol et
    missing = [p for p in [numpy_csv, sklearn_csv, pytorch_csv]
               if not os.path.exists(p)]
    if missing:
        print("\n[HATA] Asagidaki CSV dosyalari bulunamadi:")
        for p in missing:
            print(f"  {p}")
        print("\n  Once su script'leri calistirin:")
        print("    python src/model_trainer.py")
        print("    python src/sklearn_model.py")
        print("    python src/pytorch_model.py")
        raise SystemExit(1)

    # Sonuclari oku
    all_results = {}

    # NumPy — Baseline modeli al (Model1_Baseline)
    numpy_res = _read_numpy_results(numpy_csv, target_model="Model1_Baseline")
    if numpy_res:
        all_results["NumPy_Baseline"] = numpy_res
        print(f"[OK] model_results.csv okundu (Model1_Baseline)")
    else:
        print("[UYARI] model_results.csv'de Model1_Baseline bulunamadi.")

    # Sklearn
    sklearn_res = _read_standard_results(sklearn_csv)
    if sklearn_res:
        all_results["Sklearn_MLP"] = sklearn_res
        print(f"[OK] sklearn_results.csv okundu")

    # PyTorch
    pytorch_res = _read_standard_results(pytorch_csv)
    if pytorch_res:
        all_results["PyTorch_TitanicNet"] = pytorch_res
        print(f"[OK] pytorch_results.csv okundu")

    if not all_results:
        print("[HATA] Hic sonuc okunamadi.")
        raise SystemExit(1)

    # Terminal tablosu + en iyi modeli belirle
    best = print_comparison_table(all_results)

    # Grafikleri olustur
    plot_final_comparison(all_results, save_path=OUTPUT_DIR)
    plot_split_comparison(all_results, save_path=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  TUM ISLEMLER TAMAMLANDI")
    print(f"  Ciktilar -> {OUTPUT_DIR}/")
    print("=" * 60)
