# YZM304 Derin Öğrenme — I. Proje Ödevi
## Titanic İkili Sınıflandırma

| Alan | Bilgi |
|------|-------|
| **Ad Soyad** | [Furkan Günbaz] |
| **Öğrenci No** | [23291408] |
| **Üniversite** | Ankara Üniversitesi, Mühendislik Fakültesi |
| **Ders** | YZM304 — Derin Öğrenme |
| **Dönem** | 2025-2026 Bahar |
| **Tarih** | 2 Nisan 2026 |
| **GitHub** | [https://github.com/gunbaz/YZM304_Lab] |

---

## İçindekiler

1. [Giriş (Introduction)](#1-giriş-introduction)
2. [Yöntemler (Methods)](#2-yöntemler-methods)
   - 2.1 [Veri Seti ve Ön İşleme](#21-veri-seti-ve-ön-i̇şleme)
   - 2.2 [Veri Bölme](#22-veri-bölme)
   - 2.3 [Model Mimarisi](#23-model-mimarisi)
   - 2.4 [Hiperparametreler](#24-hiperparametreler)
   - 2.5 [Geliştirilen Modeller](#25-geliştirilen-modeller)
   - 2.6 [Kurulum ve Çalıştırma](#26-kurulum-ve-çalıştırma)
3. [Sonuçlar (Results)](#3-sonuçlar-results)
4. [Tartışma (Discussion)](#4-tartışma-discussion)
5. [Proje Yapısı](#5-proje-yapısı)

---

## 1. Giriş (Introduction)

Bu proje, **Kaggle Titanic** veri setini kullanarak yolcuların hayatta kalıp
kalmadığını tahmin eden bir ikili sınıflandırma sistemi geliştirmeyi amaçlamaktadır.
Hedef değişken `Survived` (0 = hayatta kalmadı, 1 = hayatta kaldı) olmak üzere
891 kayıtlık eğitim verisi üzerinde çalışılmıştır.

Çalışmanın temel katkısı, **aynı mimariyi üç farklı çerçevede** gerçekleştirerek
sonuçları doğrudan karşılaştırmaktır:

| Çerçeve | Açıklama |
|---------|----------|
| **Saf NumPy** | Sinir ağı sıfırdan, herhangi bir ML kütüphanesi kullanılmadan |
| **Scikit-learn** | `MLPClassifier` ile aynı mimari |
| **PyTorch** | `nn.Module` tabanlı `TitanicNet` |

Bu yaklaşım; ileri yayılım, geri yayılım ve ağırlık güncelleme mekanizmalarını
matematiksel düzeyde anlamayı güçlendirmekte, aynı zamanda farklı çerçevelerin
pratik kullanımını karşılaştırma imkânı sunmaktadır.

---

## 2. Yöntemler (Methods)

### 2.1 Veri Seti ve Ön İşleme

**Veri kaynağı:** Kaggle Titanic — Train Set (`train.csv`)  
**Boyut:** 891 satır × 12 sütun (ham) → işleme sonrası 22 özellik

#### Eksik Değer Stratejileri

| Sütun | Eksik Oran | Uygulanan Strateji |
|-------|-----------|-------------------|
| `Age` | ~19.9% | Pclass + Sex grubuna göre medyan ile doldurma |
| `Embarked` | ~0.2% | En sık değer (mod) ile doldurma |
| `Cabin` | ~77.1% | İkili özelliğe (`Has_Cabin`: 1/0) dönüştürme |

#### Özellik Mühendisliği

| Yeni Özellik | Tanım | Tür |
|-------------|-------|-----|
| `FamilySize` | SibSp + Parch + 1 | Sayısal |
| `IsAlone` | FamilySize == 1 ise 1, değilse 0 | İkili |
| `Title` | İsimden regex ile çekilen unvan (Mr/Miss/Mrs/Master/Rare) | Kategorik |
| `AgeBin` | Yaşın 5 eşit aralığa bölünmüş hali | Sıralı kategorik |
| `FareBin` | Ücretin 4 çeyrekliğe (quartile) bölünmüş hali | Sıralı kategorik |
| `Age_Pclass` | Age × Pclass etkileşim özelliği | Sayısal |
| `Has_Cabin` | Kabin bilgisi var/yok | İkili |

#### Kodlama (Encoding)

| Sütun | Yöntem |
|-------|--------|
| `Sex` | Binary (male=1, female=0) |
| `Embarked` | One-hot encoding (drop_first=True) |
| `Title` | One-hot encoding (drop_first=True) |
| `AgeBin` | One-hot encoding (drop_first=True) |
| `FareBin` | One-hot encoding (drop_first=True) |

#### Ölçekleme

Sürekli sayısal sütunlar (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`,
`Age_Pclass`) **StandardScaler** ile standartlaştırılmıştır.
İkili/dummy sütunlar ölçeklenmemiştir.

---

### 2.2 Veri Bölme

Sınıf dengesi korunarak **stratified** bölme uygulanmıştır:

| Küme | Oran | Satır Sayısı | Boyut |
|------|------|-------------|-------|
| Train | %70 | 623 | (623, 22) |
| Dev (Doğrulama) | %15 | 134 | (134, 22) |
| Test | %15 | 134 | (134, 22) |

`random_state=42` tüm modellerde sabittir; böylece tüm karşılaştırmalar aynı
veri bölmesi üzerinde yapılmıştır.

---

### 2.3 Model Mimarisi

Üç çerçevedeki tüm modeller **aynı temel mimariyi** paylaşmaktadır:

```
Giriş(22)  →  Gizli(16, sigmoid)  →  Çıkış(1, sigmoid)
```

| Katman | Giriş Boyutu | Çıkış Boyutu | Aktivasyon |
|--------|-------------|-------------|-----------|
| Gizli Katman (fc1) | 22 | 16 | Sigmoid |
| Çıkış Katmanı (fc2) | 16 | 1 | Sigmoid |

**Kayıp Fonksiyonu:** Binary Cross Entropy (BCE)

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Ağırlık Başlatma:** Xavier (Glorot) — `std = √(2 / (fan_in + fan_out))`

---

### 2.4 Hiperparametreler

| Hiperparametre | Değer | Açıklama |
|---------------|-------|---------|
| `learning_rate` | 0.01 | SGD adım büyüklüğü |
| `hidden_size` | 16 | Gizli katman nöron sayısı |
| `n_steps / n_epochs` | 10 000 | Toplam eğitim adımı |
| `optimizer` | SGD | Stochastic Gradient Descent |
| `loss` | Binary Cross Entropy | İkili sınıflandırma kaybı |
| `weight_init` | Xavier | Vanishing gradient'i azaltır |
| `threshold` | 0.5 | Sigmoid → sınıf (0/1) eşiği |
| `random_state` | 42 | Tekrar üretilebilirlik |
| `train / dev / test` | 0.70 / 0.15 / 0.15 | Stratified bölme oranları |
| `lambda_l2` (Model 4) | 0.01 | L2 regularization katsayısı |
| `dropout_rate` (Model 5) | 0.3 | Dropout olasılığı |

---

### 2.5 Geliştirilen Modeller

| # | Model Adı | Mimari | Özellik |
|---|-----------|--------|---------|
| 1 | **Baseline** | 22→16→1 | Temel model (lab referansı) |
| 2 | **Derin (DeepNN)** | 22→16→16→1 | Ekstra gizli katman (3 katman) |
| 3 | **Geniş** | 22→64→1 | Büyük gizli katman |
| 4 | **L2 Reg** | 22→16→1 + L2 | Weight decay (λ=0.01) |
| 5 | **Dropout** | 22→32→1 + Dropout | p=0.3; inverted dropout |

Her model saf NumPy ile yazılmıştır; `NeuralNetwork` temel sınıfından
kalıtım (inheritance) yoluyla genişletilmiştir.

---

### 2.6 Kurulum ve Çalıştırma

#### Gereksinimler

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

#### Adım Adım Çalıştırma

```bash
# 1. Veriyi hazırla ve EDA grafiklerini üret
python src/data_preprocessing.py

# 2. Saf NumPy sinir ağını eğit (Baseline)
python src/neural_network.py

# 3. 5 NumPy modelini karşılaştır
python src/model_trainer.py

# 4. Scikit-learn modelini eğit
python src/sklearn_model.py

# 5. PyTorch modelini eğit
python src/pytorch_model.py

# 6. Üç çerçeveyi karşılaştır
python src/compare_all.py
```

> **Not:** `data/raw/titanic.csv` dosyasının mevcut olduğundan emin olun.
> Yoksa [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
> sayfasından indirip `data/raw/` klasörüne kopyalayın.

---

## 3. Sonuçlar (Results)

### 3.1 NumPy Modelleri Karşılaştırması

Tüm modeller aynı veri bölmesi üzerinde eğitilmiş ve değerlendirilmiştir.

| Model | Train Acc | Dev Acc | Test Acc | Train F1 | Test F1 | Overfitting Durumu |
|-------|----------:|--------:|---------:|---------:|--------:|-------------------|
| Model1 — Baseline | 0.8202 | 0.8582 | 0.7687 | 0.7597 | 0.6854 | Normal |
| Model2 — Derin | 0.8202 | 0.8582 | 0.7687 | 0.7597 | 0.6854 | Normal |
| Model3 — Geniş | 0.8202 | 0.8582 | 0.7687 | 0.7597 | 0.6854 | Normal |
| Model4 — L2 Reg | 0.8331 | 0.8507 | 0.7910 | 0.7679 | 0.6957 | Normal |
| Model5 — Dropout | 0.8331 | 0.8507 | 0.7910 | 0.7679 | 0.6957 | Normal |

> Train – Dev accuracy farkı 0.10'u aşmayan modeller "Normal" olarak
> sınıflandırılmıştır.

### 3.2 Üç Çerçeve Karşılaştırması (Test Seti)

| Çerçeve | Test Acc | Test Precision | Test Recall | Test F1 |
|---------|--------:|---------------:|------------:|--------:|
| **NumPy Baseline** | 0.7687 | — | — | 0.6854 |
| **Sklearn MLP** | 0.7537 | 0.6800 | 0.6667 | 0.6733 |
| **PyTorch TitanicNet** | 0.7687 | 0.7273 | 0.6275 | 0.6737 |

*NumPy Baseline modelinde Precision ve Recall hesaplanmamıştır; karşılaştırma accuracy ve F1 üzerinden yapılmıştır.*

#### Eğitim (Train) ve Doğrulama (Dev) Performansları

| Çerçeve | Train Acc | Train F1 | Dev Acc | Dev F1 |
|---------|----------:|---------:|--------:|-------:|
| Sklearn MLP | 0.9165 | 0.8874 | 0.8134 | 0.7423 |
| PyTorch TitanicNet | 0.8250 | 0.7666 | 0.8582 | 0.8119 |

### 3.3 En İyi Model

**Test accuracy** ölçütüne göre **NumPy Baseline** ve **PyTorch TitanicNet**
modelleri eşit performans sergilemiştir (%76.87).

F1 skoru açısından değerlendirildiğinde PyTorch modeli (0.6737) Sklearn (0.6733)
modelini hafifçe geçmiştir. Dev/Test tutarlılığı açısından **PyTorch TitanicNet**
en dengeli sonuçları vermiştir (Dev F1: 0.8119).

### 3.4 Üretilen Grafikler

| Dosya | Açıklama |
|-------|----------|
| `outputs/eda/eda_survival_distribution.png` | Hayatta kalma dağılımı (genel, cinsiyet, sınıf) |
| `outputs/eda/eda_missing_values.png` | Sütun bazında eksik değer oranları |
| `outputs/eda/eda_feature_distributions.png` | Age, Fare, SibSp, Parch histogram + boxplot |
| `outputs/eda/eda_correlation_matrix.png` | Sayısal özellikler korelasyon ısı haritası |
| `outputs/learning_curve.png` | NumPy Baseline train/dev loss eğrisi |
| `outputs/model_comparison_loss.png` | 5 NumPy modelinin train ve dev loss eğrileri |
| `outputs/model_comparison_accuracy.png` | 5 NumPy modelinin accuracy bar chart'ı |
| `outputs/sklearn_confusion_matrix.png` | Sklearn test seti confusion matrix |
| `outputs/sklearn_metrics.png` | Sklearn train/dev/test metrik bar chart'ı |
| `outputs/pytorch_confusion_matrix.png` | PyTorch test seti confusion matrix |
| `outputs/pytorch_learning_curve.png` | PyTorch train/dev loss eğrisi |
| `outputs/pytorch_metrics.png` | PyTorch train/dev/test metrik bar chart'ı |
| `outputs/final_comparison.png` | **Üç çerçevenin nihai test metrik karşılaştırması** |
| `outputs/split_comparison.png` | Train/Dev/Test accuracy — model bazında 3 panel |

---

## 4. Tartışma (Discussion)

### 4.1 Sonuçların Yorumu

Tüm modeller benzer test accuracy değerleri (~%75–79) elde etmiştir. Bu tutarlılık,
belirlenen mimarinin bu veri seti için makul bir tavan oluşturduğuna işaret etmektedir.

- **NumPy Baseline** ve **PyTorch TitanicNet** en yüksek test accuracy
  değerini (%76.87) paylaşmaktadır.
- **Sklearn MLP**, eğitimde en yüksek accuracy'ye (%91.65) ulaşmış ancak
  test setinde %75.37'ye gerilemiştir. Bu durum sklearn'in iç optimizasyonu
  sırasında eğitim verisine aşırı uyum sağladığına işaret etmektedir.
- **PyTorch** modeli dev setinde en yüksek F1 (0.8119) değerini vermiş;
  bu da sınıf dengesini en iyi yakalayan modelin PyTorch olduğunu göstermektedir.

### 4.2 Overfitting / Underfitting Gözlemleri

- **Sklearn modelinde** train accuracy (0.916) ile dev accuracy (0.813) arasındaki
  ~10.3 puanlık fark hafif overfitting işaretidir. Bu durum, kütüphanenin
  varsayılan iç optimizasyonunun (Adam benzeri momentum) saf SGD'ye kıyasla
  daha agresif öğrenmesinden kaynaklanabilir.
- **NumPy ve PyTorch** modellerinde train–dev farkı 0.04–0.05 aralığında
  kalmıştır; overfitting riski düşük olarak değerlendirilebilir.
- Hiçbir modelde underfitting (train accuracy < 0.75) gözlemlenmemiştir.

### 4.3 Regularizasyon ve Dropout'un Etkisi

- **Model 4 (L2 Reg)** ve **Model 5 (Dropout)**, Baseline modele kıyasla
  test accuracy'de yaklaşık 2.2 puanlık iyileşme sağlamıştır (0.7687 → 0.7910).
  Bu sonuç, regularizasyonun küçük veri setlerinde genelleme yeteneğini
  anlamlı biçimde artırdığını doğrulamaktadır.
- L2 regularization büyük ağırlıkları cezalandırarak daha pürüzsüz bir karar
  sınırı oluşturmuştur. Dropout ise her eğitim adımında rastgele nöronları
  maskeleyerek modeli birden fazla alt ağ üzerinde eğitmiş ve
  ortalama etkisi itibariyle bir ensemble davranışı sergilemiştir.

### 4.4 NumPy / Sklearn / PyTorch Karşılaştırması

| Kriter | NumPy | Sklearn | PyTorch |
|--------|-------|---------|---------|
| **Esneklik** | Tam kontrol | Sınırlı | Yüksek |
| **Kod karmaşıklığı** | Yüksek | Düşük | Orta |
| **Eğitim hızı** | Yavaş | Hızlı | Orta |
| **GPU desteği** | Yok | Yok | Var |
| **Öğrenme değeri** | Çok yüksek | Düşük | Yüksek |
| **Test Accuracy** | %76.87 | %75.37 | %76.87 |

Saf NumPy implementasyonu, ileri/geri yayılım mekanizmalarını en şeffaf
biçimde ortaya koyması açısından en yüksek öğrenme değerine sahiptir.
PyTorch, sezgisel API'si ve GPU desteğiyle büyük ölçekli problemlerde
tercih edilmesi gereken çerçevedir.

### 4.5 Kısıtlamalar

1. **Veri boyutu:** 891 kayıt nispeten küçüktür; daha büyük bir veri seti
   farklı mimariler arasındaki farkları daha belirginleştirebilir.
2. **Hiperparametre optimizasyonu:** Öğrenme hızı, gizli katman boyutu ve
   adım sayısı sabit tutulmuştur. Grid search veya Bayesian optimizasyonu
   ile daha iyi sonuçlar elde edilebilir.
3. **Batch eğitim:** Tüm modeller tam-batch (full-batch) SGD ile eğitilmiştir.
   Mini-batch yaklaşımı hem eğitim hızını artırabilir hem de optimizasyona
   düzgünsüzlük (stochasticity) kazandırabilir.
4. **Tek çalıştırma:** Sonuçlar tek bir `random_state=42` tohumu ile elde
   edilmiştir; istatistiksel güvenilirlik için çoklu çalıştırma gereklidir.
5. **Veri dengesizliği:** Veri setinde %61.6 (hayatta kalmadı) /
   %38.4 (hayatta kaldı) oranı mevcuttur; SMOTE veya sınıf ağırlıklandırması
   dengesizliği azaltabilir.

### 4.6 Gelecek Çalışmalar

- **Adam Optimizer:** SGD yerine Adam kullanımı daha hızlı yakınsama ve
  daha yüksek doğruluk sağlayabilir.
- **Batch Normalization:** Derin modellerde (Model 2) eğitimin istikrarını
  ve hızını artırabilir.
- **Daha Derin Ağlar:** 3–4 gizli katmanlı değerlendirmeler mimarinin
  tavanını test edecektir.
- **Hiperparametre Araması:** Optuna veya scikit-learn `GridSearchCV` ile
  sistematik optimizasyon yapılabilir.
- **Veri Artırma (Augmentation):** Titanic veri seti sınırlı olduğundan
  SMOTE ile sentetik örnekler oluşturulabilir.
- **K-Fold Çapraz Doğrulama:** Tek dev/test bölmesi yerine k-fold yaklaşımı
  daha güvenilir performans tahmini sunar.
- **Farklı Mimariler:** LSTM veya Transformer tabanlı modeller özellik
  etkileşimlerini farklı biçimde modelleyebilir.

---

## 5. Proje Yapısı

```
titanic_project/
├── data/
│   ├── raw/
│   │   └── titanic.csv              ← Ham veri (Kaggle)
│   └── processed/
│       ├── titanic_processed.csv    ← İşlenmiş tam veri
│       ├── X_train.npy              ← Eğitim özellikleri  (623, 22)
│       ├── y_train.npy              ← Eğitim etiketleri   (623,)
│       ├── X_dev.npy                ← Doğrulama öz.       (134, 22)
│       ├── y_dev.npy                ← Doğrulama etik.     (134,)
│       ├── X_test.npy               ← Test özellikleri    (134, 22)
│       ├── y_test.npy               ← Test etiketleri     (134,)
│       └── feature_names.csv        ← 22 özellik ismi
│
├── src/
│   ├── data_preprocessing.py        ← EDA + ön işleme pipeline (TitanicDataProcessor)
│   ├── neural_network.py            ← Saf NumPy sinir ağı (NeuralNetwork)
│   ├── model_trainer.py             ← 5 model karşılaştırma (ModelTrainer)
│   │                                   ├── DeepNeuralNetwork
│   │                                   ├── RegularizedNN
│   │                                   └── DropoutNN
│   ├── sklearn_model.py             ← Sklearn MLPClassifier (SklearnModel)
│   ├── pytorch_model.py             ← PyTorch modeli (TitanicNet + PyTorchModel)
│   └── compare_all.py               ← Üç çerçeve nihai karşılaştırma
│
├── outputs/
│   ├── eda/                         ← 4 EDA grafiği (PNG)
│   ├── learning_curve.png
│   ├── model_comparison_loss.png
│   ├── model_comparison_accuracy.png
│   ├── sklearn_confusion_matrix.png
│   ├── sklearn_metrics.png
│   ├── pytorch_confusion_matrix.png
│   ├── pytorch_learning_curve.png
│   ├── pytorch_metrics.png
│   ├── final_comparison.png         ← Nihai karşılaştırma
│   ├── split_comparison.png
│   ├── model_results.csv            ← 5 NumPy model sonuçları
│   ├── sklearn_results.csv          ← Sklearn sonuçları
│   ├── pytorch_results.csv          ← PyTorch sonuçları
│   └── model_weights.npz            ← Eğitilmiş NumPy ağırlıkları
│
└── README.md                        ← Bu dosya
```

---

## Referanslar

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
2. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
5. Kaggle. (2012). Titanic: Machine Learning from Disaster. https://www.kaggle.com/competitions/titanic

---

*Bu proje YZM304 Derin Öğrenme dersi kapsamında akademik amaçla geliştirilmiştir.*
