import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, test_loader,
                num_epochs=20, lr=0.001, device='cpu', model_name='Model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    model.to(device)

    print(f'\n{"="*65}')
    print(f'  {model_name} eğitimi başladı  |  LR={lr}  |  Epochs={num_epochs}')
    print(f'{"="*65}')

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  Epoch [{epoch+1:3d}/{num_epochs}]  '
                  f'Train: loss={tr_loss:.4f} acc={tr_acc:.2f}%  |  '
                  f'Test: loss={te_loss:.4f} acc={te_acc:.2f}%')

    print(f'\n  >> Final Test Accuracy: {history["test_acc"][-1]:.2f}%')
    return history


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_training_curves(history, model_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs, history['test_loss'],  'r-o', markersize=3, label='Test Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} – Loss Curves'); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'],  'r-o', markersize=3, label='Test Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} – Accuracy Curves'); ax2.legend(); ax2.grid(True)

    plt.suptitle(model_name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(save_dir, f'{model_name.replace(" ", "_")}_curves.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Eğitim eğrileri kaydedildi: {fname}')


def plot_confusion_matrix(all_labels, all_preds, class_names, model_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} – Confusion Matrix')
    plt.ylabel('Gerçek Etiket'); plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()

    fname = os.path.join(save_dir, f'{model_name.replace(" ", "_")}_confusion.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Karmaşıklık matrisi kaydedildi: {fname}')
    return cm


def print_report(all_labels, all_preds, class_names, model_name):
    print(f'\n  {model_name} – Sınıflandırma Raporu:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print(f'  Genel Doğruluk: {accuracy_score(all_labels, all_preds)*100:.2f}%')
