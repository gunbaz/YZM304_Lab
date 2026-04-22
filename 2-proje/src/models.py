import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Model 1: LeNet-5 tarzı CNN (CIFAR-10 için uyarlanmış, giriş: 3x32x32, çıkış: 10 sınıf).
    Katmanlar: Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> FC x3
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)   # 3x32x32 -> 6x28x28
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)                      # -> 6x14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # -> 16x10x10 -> 16x5x5
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)          # Flatten: 16*5*5 = 400
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet5Improved(nn.Module):
    """
    Model 2: Model 1 ile aynı hiperparametreler; ek olarak BatchNorm ve Dropout eklendi.
    BatchNorm: Eğitimi stabilize eder, daha hızlı yakınsama sağlar.
    Dropout(0.5): Aşırı öğrenmeyi (overfitting) azaltır.
    """
    def __init__(self, num_classes=10):
        super(LeNet5Improved, self).__init__()
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.bn1     = nn.BatchNorm2d(6)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2   = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2     = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(16 * 5 * 5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
