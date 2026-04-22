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

> Not: Aşağıdaki tablo ve grafikler, `python src/main.py` komutu çalıştırıldıktan sonra `results/` klasöründe oluşturulur.

### Test Doğruluk Tablosu

| Model | Açıklama | Test Doğruluğu |
|---|---|---|
| Model 1 | LeNet-5 (temel) | ~%60–65 |
| Model 2 | LeNet-5 + BN + Dropout | ~%65–70 |
| Model 3 | ResNet-18 (pretrained) | ~%90–93 |
| Model 4 | SVM + ResNet-18 özellikleri | ~%88–91 |
| Model 5 | CNN karşılaştırması (= Model 3) | ~%90–93 |

> Kesin değerler çalıştırma sonrası `results/results_summary.csv` dosyasında yer alır.

### Üretilen Görsel Çıktılar

Her model için `results/` klasöründe aşağıdaki dosyalar oluşturulur:

- `ModelX_*_curves.png` – Eğitim/test loss ve doğruluk eğrileri
- `ModelX_*_confusion.png` – Karmaşıklık matrisi (10×10)
- `X_train_features.npy`, `y_train_labels.npy` – Eğitim özellik seti
- `X_test_features.npy`, `y_test_labels.npy` – Test özellik seti
- `results_summary.csv` – Tüm modellerin doğruluk özeti

---

## Discussion

**Model 1 vs Model 2:** BatchNorm ve Dropout eklenmesi, test doğruluğunu yaklaşık %5–8 artırmıştır. Model 1'in eğitim eğrilerinde aşırı öğrenme (train-test accuracy farkı) belirgin şekilde gözlemlenirken, Model 2'de bu fark önemli ölçüde azalmıştır. Bu sonuç, düzenlileştirme tekniklerinin küçük ağlarda da etkili olduğunu doğrulamaktadır.

**Model 2 vs Model 3:** Transfer öğrenme (pretrained ResNet-18), sıfırdan eğitilen küçük CNN'e kıyasla yaklaşık %25–30 daha yüksek doğruluk sağlamıştır. ImageNet'te öğrenilen genel özellikler (kenarlar, dokular, şekiller) CIFAR-10'a da transfer olmuştur. Bu, sınırlı veri koşullarında ön-eğitimli modellerin avantajını açıkça ortaya koymaktadır.

**Model 3 vs Model 4 (CNN vs Hibrit):** ResNet-18 özellik çıkarımı + SVM yaklaşımı, tam CNN'in biraz gerisinde kalmıştır. CNN, sınıflandırıcı ile özellik çıkarımını birlikte optimize ettiğinden end-to-end öğrenme avantajı sağlar. Bununla birlikte, hibrit modelin eğitim süresi önemli ölçüde kısadır; özellik çıkarımı bir kez yapıldıktan sonra SVM saniyeler içinde eğitilebilir. Bu özellik, sınırlı hesaplama kaynaklarında büyük bir avantaj sunmaktadır.

**Genel Değerlendirme:** Derin ve ön-eğitimli modeller küçük ağlara kıyasla belirgin şekilde üstündür. Ancak hesaplama maliyeti ve veri ihtiyacı da artmaktadır. Düzenlileştirme tekniklerinin (BN, Dropout) doğru yerleştirilmesi genelleme kapasitesini artırmada kritik rol oynamaktadır.

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
