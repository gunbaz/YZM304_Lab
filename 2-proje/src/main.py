"""
YZM304 Derin Öğrenme – II. Proje Ödevi
CNN ile Özellik Çıkarma ve Sınıflandırma (CIFAR-10)

Modeller:
  Model 1 – LeNet-5 tarzı CNN (temel)
  Model 2 – LeNet-5 + BatchNorm + Dropout (iyileştirilmiş)
  Model 3 – ResNet-18 (torchvision, pretrained=True, fine-tuned)
  Model 4 – Hibrit: ResNet-18 özellik çıkarımı + SVM
  Model 5 – Model 3 (ResNet-18) CNN karşılaştırması olarak kullanılır
             (Ödev notu: ilk 3 modelden biri seçilirse ayrıca Model 5 gerekmez)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Proje kök dizininden çalıştırılabilmesi için src klasörünü ekle
sys.path.insert(0, os.path.dirname(__file__))

from models import LeNet5, LeNet5Improved
from train_eval import (
    CIFAR10_CLASSES, train_model, get_predictions,
    plot_training_curves, plot_confusion_matrix, print_report
)

# ──────────────────────────────────────────────────────────────
# Yapılandırma
# ──────────────────────────────────────────────────────────────
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE  = 64
EPOCHS_1_2  = 20       # Model 1 ve 2 için epoch sayısı
EPOCHS_3    = 10       # Model 3 için (pretrained, daha az epoch yeterli)
LR_1_2      = 0.001   # Adam optimizer öğrenme hızı (Model 1-2)
LR_3        = 0.0001  # Fine-tuning için daha küçük lr (Model 3)
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
print(f'Cihaz: {DEVICE}')
print(f'Sonuçlar: {os.path.abspath(RESULTS_DIR)}\n')

# ──────────────────────────────────────────────────────────────
# Veri Yükleme – CIFAR-10
# ──────────────────────────────────────────────────────────────
# Ortalama ve standart sapma CIFAR-10 eğitim setinden hesaplanmış değerler
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # Veri artırma: yatay çevirme
    transforms.RandomCrop(32, padding=4),       # Veri artırma: rastgele kırpma
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

train_set = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,  download=True, transform=train_transform)
test_set  = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f'CIFAR-10 yüklendi: {len(train_set)} eğitim, {len(test_set)} test örneği')
print(f'Görüntü boyutu: 3x32x32  |  Sınıf sayısı: 10\n')

# ──────────────────────────────────────────────────────────────
# MODEL 1 – LeNet-5 tarzı temel CNN
# ──────────────────────────────────────────────────────────────
print('=' * 65)
print('MODEL 1: LeNet-5 Tarzı Temel CNN')
print('=' * 65)
model1 = LeNet5(num_classes=10)
param_count = sum(p.numel() for p in model1.parameters())
print(f'Toplam parametre sayısı: {param_count:,}')

history1 = train_model(model1, train_loader, test_loader,
                       num_epochs=EPOCHS_1_2, lr=LR_1_2,
                       device=DEVICE, model_name='Model1_LeNet5')

labels1, preds1 = get_predictions(model1, test_loader, DEVICE)
print_report(labels1, preds1, CIFAR10_CLASSES, 'Model1_LeNet5')
plot_training_curves(history1, 'Model1_LeNet5', RESULTS_DIR)
plot_confusion_matrix(labels1, preds1, CIFAR10_CLASSES, 'Model1_LeNet5', RESULTS_DIR)
torch.save(model1.state_dict(), os.path.join(RESULTS_DIR, 'model1_lenet5.pth'))

# ──────────────────────────────────────────────────────────────
# MODEL 2 – LeNet-5 + BatchNorm + Dropout
# ──────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('MODEL 2: LeNet-5 + BatchNorm + Dropout')
print('=' * 65)
model2 = LeNet5Improved(num_classes=10)
param_count2 = sum(p.numel() for p in model2.parameters())
print(f'Toplam parametre sayısı: {param_count2:,}')

history2 = train_model(model2, train_loader, test_loader,
                       num_epochs=EPOCHS_1_2, lr=LR_1_2,
                       device=DEVICE, model_name='Model2_LeNet5_Improved')

labels2, preds2 = get_predictions(model2, test_loader, DEVICE)
print_report(labels2, preds2, CIFAR10_CLASSES, 'Model2_LeNet5_Improved')
plot_training_curves(history2, 'Model2_LeNet5_Improved', RESULTS_DIR)
plot_confusion_matrix(labels2, preds2, CIFAR10_CLASSES, 'Model2_LeNet5_Improved', RESULTS_DIR)
torch.save(model2.state_dict(), os.path.join(RESULTS_DIR, 'model2_lenet5_improved.pth'))

# ──────────────────────────────────────────────────────────────
# MODEL 3 – ResNet-18 (torchvision, pretrained=True)
# CIFAR-10 için uyarlama: conv1 stride=1, maxpool kaldırıldı, fc=10
# ──────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('MODEL 3: ResNet-18 (Pretrained=True, CIFAR-10 Fine-Tuning)')
print('=' * 65)

try:
    model3 = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
except AttributeError:
    model3 = tv_models.resnet18(pretrained=True)

# CIFAR-10 32x32 için ilk katmanı uyarla (orijinal ResNet 224x224 için tasarlanmış)
model3.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model3.maxpool = nn.Identity()
model3.fc      = nn.Linear(model3.fc.in_features, 10)

param_count3 = sum(p.numel() for p in model3.parameters())
print(f'Toplam parametre sayısı: {param_count3:,}')

history3 = train_model(model3, train_loader, test_loader,
                       num_epochs=EPOCHS_3, lr=LR_3,
                       device=DEVICE, model_name='Model3_ResNet18')

labels3, preds3 = get_predictions(model3, test_loader, DEVICE)
print_report(labels3, preds3, CIFAR10_CLASSES, 'Model3_ResNet18')
plot_training_curves(history3, 'Model3_ResNet18', RESULTS_DIR)
plot_confusion_matrix(labels3, preds3, CIFAR10_CLASSES, 'Model3_ResNet18', RESULTS_DIR)
torch.save(model3.state_dict(), os.path.join(RESULTS_DIR, 'model3_resnet18.pth'))

# ──────────────────────────────────────────────────────────────
# MODEL 4 – Hibrit: CNN Özellik Çıkarımı + SVM
# Model 3 (ResNet-18) son FC katmanı çıkarılarak özellik vektörü elde edilir.
# Özellikler ve etiketler .npy dosyalarına kaydedilir, ardından SVM eğitilir.
# ──────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('MODEL 4: Hibrit – ResNet-18 Özellik Çıkarımı + SVM')
print('=' * 65)


def extract_features(model, loader, device):
    """ResNet-18'in son FC katmanını çıkararak özellik vektörlerini döndürür."""
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # avg_pool çıktısı: 512-d
    feature_extractor.eval().to(device)
    feats_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            feats = feature_extractor(inputs.to(device))
            feats_list.append(feats.view(feats.size(0), -1).cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(feats_list), np.concatenate(labels_list)


print('ResNet-18 ile özellik çıkarımı yapılıyor...')
X_train, y_train = extract_features(model3, train_loader, DEVICE)
X_test,  y_test  = extract_features(model3, test_loader,  DEVICE)

print(f'\nÖzellik seti boyutları:')
print(f'  X_train : {X_train.shape}   (örnek sayısı x özellik boyutu)')
print(f'  y_train : {y_train.shape}')
print(f'  X_test  : {X_test.shape}')
print(f'  y_test  : {y_test.shape}')
print(f'  Özellik boyutu (feature_dim): {X_train.shape[1]}')
print(f'  Eğitim örnek sayısı         : {X_train.shape[0]}')
print(f'  Test örnek sayısı           : {X_test.shape[0]}')

np.save(os.path.join(RESULTS_DIR, 'X_train_features.npy'), X_train)
np.save(os.path.join(RESULTS_DIR, 'y_train_labels.npy'),   y_train)
np.save(os.path.join(RESULTS_DIR, 'X_test_features.npy'),  X_test)
np.save(os.path.join(RESULTS_DIR, 'y_test_labels.npy'),    y_test)
print(f'\n.npy dosyaları {RESULTS_DIR}/ klasörüne kaydedildi.')

# Özellikleri normalleştir (SVM için gerekli)
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print('\nSVM (RBF kernel, C=10) eğitiliyor...')
svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
svm.fit(X_train_sc, y_train)
svm_preds = svm.predict(X_test_sc)
svm_acc   = accuracy_score(y_test, svm_preds) * 100

print(f'\nModel 4 (SVM) Test Doğruluğu: {svm_acc:.2f}%')
print('\nModel 4 (SVM) Sınıflandırma Raporu:')
print(classification_report(y_test, svm_preds, target_names=CIFAR10_CLASSES))
plot_confusion_matrix(y_test, svm_preds, CIFAR10_CLASSES, 'Model4_SVM_Features', RESULTS_DIR)

# ──────────────────────────────────────────────────────────────
# MODEL 5 – Tam CNN Karşılaştırması
# Ödev notu: İlk 3 modelden biri kullanıldığı için ayrı Model 5 gerekmez.
# Model 3 (ResNet-18) aynı CIFAR-10 veri seti üzerinde eğitilmiş tam CNN'dir.
# ──────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('MODEL 5: Tam CNN Karşılaştırması = Model 3 (ResNet-18)')
print('(Ödev kuralı: Model 3 seçildiğinden ayrı Model 5 gerekmez)')
print('=' * 65)
cnn_acc = accuracy_score(labels3, preds3) * 100
print(f'  Model 3 (ResNet-18) Test Doğruluğu : {cnn_acc:.2f}%')
print(f'  Model 4 (SVM)       Test Doğruluğu : {svm_acc:.2f}%')
print(f'  Fark (CNN - SVM)                   : {cnn_acc - svm_acc:+.2f}%')

# ──────────────────────────────────────────────────────────────
# SONUÇ TABLOSU
# ──────────────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('GENEL SONUÇ TABLOSU')
print('=' * 65)
from sklearn.metrics import accuracy_score as acc_fn

results = [
    ('Model 1  LeNet-5 (temel)',              acc_fn(labels1, preds1) * 100),
    ('Model 2  LeNet-5 + BN + Dropout',       acc_fn(labels2, preds2) * 100),
    ('Model 3  ResNet-18 (pretrained)',        acc_fn(labels3, preds3) * 100),
    ('Model 4  SVM + ResNet-18 Özellikleri',  svm_acc),
    ('Model 5  ResNet-18 CNN (= Model 3)',     acc_fn(labels3, preds3) * 100),
]

print(f'  {"Model":<42} {"Test Doğruluğu":>14}')
print('  ' + '-' * 58)
for name, score in results:
    print(f'  {name:<42} {score:>13.2f}%')

# CSV olarak kaydet
csv_path = os.path.join(RESULTS_DIR, 'results_summary.csv')
with open(csv_path, 'w') as f:
    f.write('Model,Test_Accuracy\n')
    for name, score in results:
        f.write(f'{name},{score:.2f}\n')
print(f'\n  Sonuç tablosu kaydedildi: {csv_path}')
