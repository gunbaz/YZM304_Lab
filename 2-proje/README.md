# YZM304 Derin Öğrenme – II. Proje: CNN ile Özellik Çıkarma ve Sınıflandırma

## Introduction

Evrişimli Sinir Ağları (Convolutional Neural Networks – CNN), görüntü sınıflandırma alanında günümüzde en başarılı derin öğrenme mimarilerinden birini oluşturmaktadır. LeCun vd. (1998) tarafından tanıtılan LeNet-5, modern CNN mimarilerinin öncüsü olup özellikle el yazısı rakam tanımada büyük başarı elde etmiştir. Sonraki yıllarda AlexNet (Krizhevsky vd., 2012), VGG (Simonyan & Zisserman, 2014) ve ResNet (He vd., 2016) gibi daha derin ve güçlü mimariler geliştirilmiş; bu modeller ImageNet yarışmalarında insan düzeyine yakın başarımlar sergilemiştir.

Bu projede CIFAR-10 benchmark veri seti üzerinde beş farklı model incelenmiştir:

- **Model 1** – LeNet-5 tarzı temel CNN (sıfırdan eğitim)
- **Model 2** – Model 1 + BatchNorm + Dropout (düzenlileştirilmiş)
- **Model 3** – ResNet-18 (ImageNet ön-eğitimli, CIFAR-10 ince ayarı)
- **Model 4** – Hibrit: ResNet-18 özellik çıkarımı + Destek Vektör Makinesi (SVM)
- **Model 5** – Tam CNN karşılaştırması (Model 3 olarak kabul edilmiştir)

Projenin amacı; temel CNN mimarisini baştan inşa etmek, düzenlileştirme tekniklerinin etkisini ölçmek, transfer öğrenmenin avantajlarını gözlemlemek ve hibrit bir yaklaşımın (CNN + klasik ML) performansını saf derin öğrenme ile karşılaştırmaktır.

---

## Method

### Veri Seti

**CIFAR-10** (Canadian Institute For Advanced Research – 10 sınıf): 60.000 renkli (RGB) görüntüden oluşur; 50.000 eğitim ve 10.000 test örneği içerir. Her görüntü 32×32 piksel boyutundadır. Sınıflar: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

#### Ön İşleme

Tüm modeller için aynı eğitim/test ayrımı kullanılmıştır:

| Dönüşüm | Eğitim | Test |
|---|---|---|
| RandomHorizontalFlip | ✓ | ✗ |
| RandomCrop(32, padding=4) | ✓ | ✗ |
| ToTensor | ✓ | ✓ |
| Normalize (μ=(0.491, 0.482, 0.447), σ=(0.247, 0.244, 0.262)) | ✓ | ✓ |

Normalize değerleri CIFAR-10 eğitim seti üzerinden hesaplanmış standart değerlerdir. Veri artırma (augmentation) teknikleri yalnızca eğitim setine uygulanmıştır; test seti bozulmadan değerlendirilmiştir.

---

### Model 1 – LeNet-5 Tarzı Temel CNN

LeNet-5 mimarisi, CIFAR-10'un 3 kanallı 32×32 giriş boyutuna uyarlanmıştır:

```
Giriş: 3×32×32
→ Conv2d(3→6, k=5)  → ReLU → MaxPool(2×2)   # 6×14×14
→ Conv2d(6→16, k=5) → ReLU → MaxPool(2×2)   # 16×5×5
→ Flatten (400)
→ FC(400→120) → ReLU
→ FC(120→84)  → ReLU
→ FC(84→10)
```

**Hiperparametreler:** Optimizer: Adam, LR: 0.001, Epochs: 20, Batch Size: 64.
Adam optimizer, momentum ve adaptif öğrenme hızı özelliği sayesinde SGD'ye kıyasla daha hızlı yakınsama sağlamaktadır.

---

### Model 2 – LeNet-5 + BatchNorm + Dropout

Model 1 ile aynı katman hiperparametreleri korunmuş; aşırı öğrenmeyi azaltmak amacıyla iki iyileştirme eklenmiştir:

- **Batch Normalization:** Her Conv katmanının ardından uygulanır. İç kovaryat kaymasını (internal covariate shift) azaltarak eğitimi stabilize eder ve daha yüksek öğrenme hızlarına olanak tanır (Ioffe & Szegedy, 2015).
- **Dropout (p=0.5):** FC1 ve FC2 katmanlarının ardından uygulanır. Eğitim sırasında nöronların %50'sini rastgele devre dışı bırakarak co-adaptation'ı önler ve genelleme kapasitesini artırır (Srivastava vd., 2014).

```
Giriş: 3×32×32
→ Conv2d(3→6, k=5) → BN → ReLU → MaxPool(2×2)
→ Conv2d(6→16, k=5) → BN → ReLU → MaxPool(2×2)
→ Flatten → FC(400→120) → ReLU → Dropout(0.5)
→ FC(120→84) → ReLU → Dropout(0.5)
→ FC(84→10)
```

---

### Model 3 – ResNet-18 (Pretrained, Fine-Tuning)

PyTorch `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` kullanılmıştır. ResNet-18, artık bağlantılar (residual connections) sayesinde derin ağlarda görülen gradyan kaybolması sorununu çözer (He vd., 2016).

CIFAR-10'un 32×32 giriş boyutuna uyarlama:
- `conv1`: kernel 7×7 stride 2 → kernel 3×3 stride 1 (bilgi kaybını önlemek için)
- `maxpool`: `nn.Identity()` ile devre dışı bırakıldı
- `fc`: 512 → 10 (CIFAR-10 sınıf sayısına göre)

**Hiperparametreler:** Optimizer: Adam, LR: 0.0001 (ön-eğitimli ağırlıkları bozmamak için küçük lr), Epochs: 10, Batch Size: 64.

---

### Model 4 – Hibrit: CNN Özellik Çıkarımı + SVM

Eğitilmiş Model 3'ün (ResNet-18) son FC katmanı çıkarılarak tüm eğitim ve test görüntüleri için 512 boyutlu özellik vektörleri elde edilmiştir. Bu özellikler `.npy` formatında kaydedilmiştir.

Ardından **SVM (RBF çekirdek, C=10, gamma='scale')** bu özellikler üzerinde eğitilmiştir. StandardScaler ile normalizasyon uygulanmıştır.

**Model 5 notu:** Ödev talimatına göre, ilk 3 modelden biri hibrit modelle aynı veri setinde eğitilmişse ayrıca Model 5 gerekmemektedir. Model 3 (ResNet-18) CIFAR-10 üzerinde eğitildiğinden, tam CNN karşılaştırması olarak Model 3 kullanılmıştır.

---

### Eğitim Detayları

| Parametre | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---|---|---|---|
| Loss Fonksiyonu | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss | – |
| Optimizer | Adam | Adam | Adam | – |
| Learning Rate | 0.001 | 0.001 | 0.0001 | – |
| Epoch | 20 | 20 | 10 | – |
| Batch Size | 64 | 64 | 64 | – |
| ML Modeli | – | – | – | SVM (RBF, C=10) |

---

## Results

### Test Doğruluk Tablosu

Tüm modeller CIFAR-10 test seti (10.000 örnek) üzerinde değerlendirilmiştir.

| Model | Açıklama | Test Doğruluğu |
|---|---|---|
| Model 1 | LeNet-5 (temel CNN) | **%65.56** |
| Model 2 | LeNet-5 + BatchNorm + Dropout | **%62.32** |
| Model 3 | ResNet-18 (pretrained, fine-tuned) | **%92.50** |
| Model 4 | SVM + ResNet-18 özellikleri (hibrit) | **%93.56** |
| Model 5 | Tam CNN karşılaştırması (= Model 3) | **%92.50** |

### Model 1 – Sınıf Bazında Sonuçlar (LeNet-5)

| Sınıf | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.71 | 0.70 | 0.71 |
| automobile | 0.74 | 0.84 | 0.79 |
| bird | 0.65 | 0.46 | 0.54 |
| cat | 0.50 | 0.40 | 0.45 |
| deer | 0.61 | 0.59 | 0.60 |
| dog | 0.46 | 0.69 | 0.55 |
| frog | 0.72 | 0.75 | 0.73 |
| horse | 0.64 | 0.73 | 0.68 |
| ship | 0.81 | 0.73 | 0.77 |
| truck | 0.82 | 0.66 | 0.73 |
| **Genel** | **0.67** | **0.66** | **0.65** |

### Eğitim Süreci – Model 1 Epoch Detayları

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1 | 1.7482 | %35.0 | 1.4905 | %45.8 |
| 5 | 1.2493 | %55.1 | 1.1464 | %58.5 |
| 10 | 1.1137 | %60.4 | 1.0724 | %62.0 |
| 15 | 1.0577 | %62.5 | 0.9839 | %65.1 |
| 20 | 1.0089 | %64.0 | 0.9847 | %65.6 |

### Üretilen Görsel Çıktılar

Her model için aşağıdaki çıktılar üretilmiştir:

- `Model1_LeNet5_curves.png` / `Model1_LeNet5_cm.png`
- `Model2_Improved_curves.png` / `Model2_Improved_cm.png`
- `Model3_ResNet18_curves.png` / `Model3_ResNet18_cm.png`
- `Model4_SVM_cm.png`
- `X_train_features.npy` (50000 × 512), `y_train_labels.npy` (50000,)
- `X_test_features.npy` (10000 × 512), `y_test_labels.npy` (10000,)
- `results_summary.csv`

---

## Discussion

**Model 1 vs Model 2:** Beklenilerin aksine, BatchNorm ve Dropout eklenen Model 2 (%62.32), temel Model 1'den (%65.56) daha düşük doğruluk elde etmiştir. Bu durum, Dropout(p=0.5)'ın yalnızca 62.006 parametreye sahip küçük bir ağda kapasiteyi aşırı kısıtlamasından kaynaklanmaktadır. Küçük ağlarda Dropout oranının 0.2–0.3 aralığında tutulması veya yalnızca BatchNorm kullanılması daha uygun olabilirdi. Bununla birlikte Model 2'nin eğitim eğrileri, Model 1'e kıyasla daha stabil bir kayıp düşüşü sergilemiştir; bu da BatchNorm'un eğitimi stabilize etme etkisini doğrulamaktadır.

**Model 1/2 vs Model 3:** Transfer öğrenme (pretrained ResNet-18) beklendiği gibi çok büyük bir fark yaratmıştır: Model 3, Model 1'den **%26.94** daha yüksek doğruluk elde etmiştir. ImageNet'te milyonlarca görüntüyle öğrenilen genel özellikler (kenarlar, dokular, şekiller) CIFAR-10 gibi küçük veri setlerine başarıyla aktarılmıştır. Bu bulgu, sınırlı veri ve hesaplama bütçesi koşullarında transfer öğrenmenin tartışmasız üstünlüğünü kanıtlamaktadır.

**Model 3 vs Model 4 (CNN vs Hibrit):** Model 4 (SVM + ResNet-18 özellikleri, %93.56), Model 3'ü (tam CNN, %92.50) **%1.06** oranında geride bırakmıştır. Bu beklenmedik sonuç, SVM'in 512 boyutlu RBF uzayında daha iyi bir karar sınırı bulabilmesinden kaynaklanmaktadır. ResNet-18 tarafından çıkarılan yüksek kaliteli özellikler, SVM gibi klasik bir sınıflandırıcı için de yeterince ayrıştırıcıdır. Ek olarak, hibrit modelin eğitim süresi çok daha kısadır: özellik çıkarımı bir kez yapıldıktan sonra SVM saniyeler içinde eğitilebilmektedir.

**Sınıf Bazlı Analiz:** Model 1'de en zor sınıflar cat (F1=0.45) ve bird (F1=0.54) olmuştur. Bu sınıflar birbirine görsel olarak benzer (küçük hayvanlar, benzer renkler) ve küçük ağların bu ince farkları ayırt etmesi güçtür. Buna karşın automobile (F1=0.79) ve ship (F1=0.77) gibi yapay nesneler daha yüksek başarım göstermiştir.

**Genel Değerlendirme:** Derin ve ön-eğitimli modeller, sıfırdan eğitilen küçük ağlara kıyasla belirgin şekilde üstündür. Küçük ağlarda düzenlileştirme parametrelerinin dikkatli seçilmesi gerekmektedir. Transfer öğrenme + SVM kombinasyonu ise hem yüksek doğruluk hem de düşük eğitim süresi açısından en verimli yaklaşım olarak öne çıkmaktadır.

---

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*, 25.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv:1409.1556*.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
5. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456.
6. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929–1958.
7. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical Report, University of Toronto.
8. PyTorch Documentation. (2024). torchvision.models. https://pytorch.org/vision/stable/models.html

---

## Projeyi Çalıştırma

```bash
# Gereksinimleri yükle
pip install -r requirements.txt

# Ana scripti çalıştır (CIFAR-10 otomatik indirilir)
python src/main.py
```

Sonuçlar `results/` klasöründe üretilir.
