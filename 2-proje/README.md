# YZM304 Derin Öğrenme – II. Proje: CNN ile Özellik Çıkarma ve Sınıflandırma

> **Not:** Proje teslim tarihi itibarıyla görsel grafikler (loss/accuracy eğrileri ve confusion matrix PNG dosyaları) bilgisayar arızası nedeniyle depoya eklenememiştir. Tüm sayısal sonuçlar, karmaşıklık matrisi verileri ve sınıflandırma raporları aşağıda tablo formatında sunulmuştur. Grafikler bilgisayar tamirinden sonra eklenecektir. `python src/main.py` komutu çalıştırılarak tüm görseller yeniden üretilebilir.

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

Tüm modeller CIFAR-10 test seti (10.000 örnek) üzerinde değerlendirilmiştir.

### Test Doğruluk Özeti

| Model | Açıklama | Parametre Sayısı | Test Doğruluğu |
|---|---|---|---|
| Model 1 | LeNet-5 (temel CNN) | 62.006 | **%65.56** |
| Model 2 | LeNet-5 + BatchNorm + Dropout | 62.050 | **%62.32** |
| Model 3 | ResNet-18 (pretrained, fine-tuned) | 11.173.962 | **%92.50** |
| Model 4 | SVM + ResNet-18 özellikleri (hibrit) | – | **%93.56** |
| Model 5 | Tam CNN karşılaştırması (= Model 3) | 11.173.962 | **%92.50** |

---

### Model 1 – LeNet-5 Temel CNN

#### Eğitim Süreci

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1 | 1.7482 | %35.0 | 1.4905 | %45.8 |
| 5 | 1.2493 | %55.1 | 1.1464 | %58.5 |
| 10 | 1.1137 | %60.4 | 1.0724 | %62.0 |
| 15 | 1.0577 | %62.5 | 0.9839 | %65.1 |
| 20 | 1.0089 | %64.0 | 0.9847 | **%65.6** |

#### Sınıflandırma Raporu

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

#### Karmaşıklık Matrisi

| Gerçek \ Tahmin | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | **699** | 32 | 57 | 29 | 18 | 23 | 10 | 33 | 66 | 31 |
| **automobile** | 26 | **840** | 5 | 9 | 4 | 9 | 11 | 16 | 33 | 47 |
| **bird** | 62 | 10 | **462** | 60 | 111 | 165 | 71 | 43 | 12 | 4 |
| **cat** | 17 | 9 | 47 | **403** | 63 | 307 | 77 | 53 | 12 | 12 |
| **deer** | 22 | 7 | 44 | 48 | **590** | 78 | 77 | 129 | 3 | 2 |
| **dog** | 14 | 4 | 23 | 138 | 40 | **689** | 20 | 67 | 3 | 2 |
| **frog** | 11 | 8 | 30 | 48 | 68 | 68 | **751** | 12 | 2 | 2 |
| **horse** | 18 | 5 | 21 | 33 | 50 | 131 | 7 | **727** | 2 | 6 |
| **ship** | 86 | 56 | 16 | 21 | 13 | 17 | 10 | 9 | **732** | 40 |
| **truck** | 27 | 160 | 10 | 21 | 7 | 16 | 14 | 44 | 38 | **663** |

---

### Model 2 – LeNet-5 + BatchNorm + Dropout

#### Eğitim Süreci

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1 | 1.8408 | %29.8 | 1.5615 | %43.0 |
| 5 | 1.4988 | %45.6 | 1.2632 | %54.7 |
| 10 | 1.3940 | %50.6 | 1.1744 | %58.4 |
| 15 | 1.3439 | %52.7 | 1.2233 | %57.2 |
| 20 | 1.3148 | %54.2 | 1.0788 | **%62.3** |

#### Sınıflandırma Raporu

| Sınıf | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.72 | 0.59 | 0.65 |
| automobile | 0.76 | 0.79 | 0.78 |
| bird | 0.53 | 0.47 | 0.49 |
| cat | 0.37 | 0.47 | 0.42 |
| deer | 0.65 | 0.46 | 0.54 |
| dog | 0.51 | 0.49 | 0.50 |
| frog | 0.64 | 0.77 | 0.70 |
| horse | 0.69 | 0.67 | 0.68 |
| ship | 0.69 | 0.78 | 0.73 |
| truck | 0.74 | 0.74 | 0.74 |
| **Genel** | **0.63** | **0.62** | **0.62** |

#### Karmaşıklık Matrisi

| Gerçek \ Tahmin | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | **595** | 38 | 85 | 32 | 11 | 10 | 11 | 23 | 155 | 40 |
| **automobile** | 23 | **791** | 0 | 17 | 2 | 5 | 14 | 5 | 35 | 108 |
| **bird** | 63 | 10 | **465** | 116 | 84 | 88 | 93 | 45 | 21 | 15 |
| **cat** | 8 | 6 | 76 | **474** | 36 | 216 | 106 | 35 | 30 | 13 |
| **deer** | 29 | 7 | 74 | 106 | **462** | 32 | 134 | 125 | 26 | 5 |
| **dog** | 6 | 5 | 76 | 279 | 28 | **492** | 38 | 55 | 15 | 6 |
| **frog** | 3 | 8 | 50 | 110 | 31 | 13 | **768** | 4 | 9 | 4 |
| **horse** | 17 | 3 | 33 | 88 | 55 | 98 | 14 | **665** | 4 | 23 |
| **ship** | 59 | 58 | 17 | 29 | 2 | 4 | 7 | 3 | **777** | 44 |
| **truck** | 21 | 115 | 6 | 31 | 3 | 3 | 13 | 10 | 55 | **743** |

---

### Model 3 – ResNet-18 (Pretrained, Fine-Tuning)

#### Eğitim Süreci

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|---|---|---|---|---|
| 1 | 0.8167 | %71.6 | 0.4773 | %83.9 |
| 5 | 0.1865 | %93.6 | 0.2531 | %91.5 |
| 10 | 0.0881 | %96.9 | 0.2521 | **%92.5** |

#### Sınıflandırma Raporu

| Sınıf | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.91 | 0.95 | 0.93 |
| automobile | 0.96 | 0.96 | 0.96 |
| bird | 0.95 | 0.88 | 0.91 |
| cat | 0.88 | 0.77 | 0.82 |
| deer | 0.90 | 0.95 | 0.93 |
| dog | 0.83 | 0.92 | 0.87 |
| frog | 0.95 | 0.95 | 0.95 |
| horse | 0.95 | 0.95 | 0.95 |
| ship | 0.95 | 0.96 | 0.96 |
| truck | 0.96 | 0.95 | 0.95 |
| **Genel** | **0.93** | **0.92** | **0.92** |

#### Karmaşıklık Matrisi

| Gerçek \ Tahmin | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | **955** | 3 | 8 | 3 | 2 | 0 | 0 | 1 | 17 | 11 |
| **automobile** | 6 | **962** | 1 | 1 | 0 | 0 | 1 | 1 | 10 | 18 |
| **bird** | 24 | 0 | **881** | 20 | 33 | 20 | 11 | 9 | 1 | 1 |
| **cat** | 13 | 3 | 13 | **773** | 27 | 127 | 22 | 13 | 5 | 4 |
| **deer** | 4 | 0 | 4 | 12 | **952** | 11 | 5 | 12 | 0 | 0 |
| **dog** | 5 | 0 | 7 | 41 | 17 | **915** | 8 | 6 | 0 | 1 |
| **frog** | 5 | 1 | 9 | 15 | 7 | 8 | **951** | 3 | 0 | 1 |
| **horse** | 4 | 0 | 3 | 5 | 12 | 23 | 0 | **951** | 1 | 1 |
| **ship** | 25 | 3 | 2 | 0 | 2 | 0 | 2 | 0 | **963** | 3 |
| **truck** | 6 | 30 | 1 | 4 | 0 | 0 | 0 | 0 | 12 | **947** |

---

### Model 4 – Hibrit: SVM + ResNet-18 Özellikleri

**Özellik Seti Boyutları:**

| | Boyut | Açıklama |
|---|---|---|
| X_train | (50.000 × 512) | 50.000 eğitim örneği, 512 boyutlu özellik vektörü |
| y_train | (50.000,) | Eğitim etiketleri |
| X_test | (10.000 × 512) | 10.000 test örneği, 512 boyutlu özellik vektörü |
| y_test | (10.000,) | Test etiketleri |

#### Sınıflandırma Raporu

| Sınıf | Precision | Recall | F1-Score |
|---|---|---|---|
| airplane | 0.94 | 0.95 | 0.94 |
| automobile | 0.96 | 0.97 | 0.97 |
| bird | 0.93 | 0.92 | 0.93 |
| cat | 0.86 | 0.86 | 0.86 |
| deer | 0.94 | 0.94 | 0.94 |
| dog | 0.88 | 0.89 | 0.89 |
| frog | 0.96 | 0.95 | 0.96 |
| horse | 0.95 | 0.95 | 0.95 |
| ship | 0.97 | 0.96 | 0.96 |
| truck | 0.96 | 0.95 | 0.95 |
| **Genel** | **0.94** | **0.94** | **0.94** |

#### Karmaşıklık Matrisi

| Gerçek \ Tahmin | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| **airplane** | **950** | 5 | 10 | 5 | 2 | 1 | 0 | 2 | 18 | 7 |
| **automobile** | 4 | **972** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 22 |
| **bird** | 13 | 0 | **922** | 15 | 14 | 17 | 9 | 9 | 1 | 0 |
| **cat** | 6 | 1 | 15 | **860** | 16 | 71 | 15 | 7 | 3 | 6 |
| **deer** | 4 | 0 | 10 | 16 | **940** | 10 | 6 | 14 | 0 | 0 |
| **dog** | 4 | 0 | 9 | 66 | 12 | **889** | 7 | 11 | 1 | 1 |
| **frog** | 2 | 2 | 13 | 17 | 5 | 4 | **954** | 3 | 0 | 0 |
| **horse** | 4 | 0 | 4 | 12 | 13 | 13 | 0 | **952** | 1 | 1 |
| **ship** | 22 | 3 | 4 | 0 | 0 | 0 | 1 | 0 | **964** | 6 |
| **truck** | 7 | 26 | 1 | 4 | 0 | 0 | 0 | 0 | 9 | **953** |



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
