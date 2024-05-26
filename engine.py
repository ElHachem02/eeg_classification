import sys
import time
import copy
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from timm.utils import NativeScaler
from model import EEGTransformer
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix

nb_epochs = 15

def prepare_training():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = EEGTransformer().to(device)
    
    optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=1e-5)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=nb_epochs,
        lr_min=1e-6,
        warmup_t=5,
        warmup_lr_init=1e-4,
    )

    class_weights = torch.tensor([1.0, 2.0, 3.0], device=device)  # Adjust these weights based on your class distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss_scaler = NativeScaler()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, loss_scaler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, val_losses = [], []
    train_balanced_accs, val_balanced_accs = [], []

    for epoch in range(nb_epochs):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, nb_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            
            all_labels = []
            all_preds = []

            for batch in tqdm.tqdm(dataloaders[phase]):
                inputs = batch['X'].to(device)
                labels = batch['y'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                dataset_size += inputs.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == 'train':
                scheduler.step(epoch=epoch)
                
            dataset_size = float(dataset_size)  # Convert to float for division

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.float() / dataset_size

            # Compute additional metrics
            f1 = f1_score(all_labels, all_preds, average='weighted')
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            report = classification_report(all_labels, all_preds)

            if epoch%4==0:
                print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Balanced Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, f1, balanced_acc))
                print(report)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_balanced_accs.append(balanced_acc)
            else:
                val_losses.append(epoch_loss)
                val_balanced_accs.append(balanced_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    epochs = range(1, nb_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Convert lists to numpy arrays for plotting
    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    train_balanced_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_balanced_accs]
    val_balanced_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_balanced_accs]

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_balanced_accs, label='Training Balanced Accuracy')
    plt.plot(epochs, val_balanced_accs, label='Validation Balanced Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.title('Training and Validation Balanced Accuracy')
    plt.legend()

    plt.show()

    # Evaluate on the test set
    model.eval()
    test_running_loss = 0.0
    test_running_corrects = 0
    test_dataset_size = 0
    
    test_labels = []
    test_preds = []

    for batch in tqdm.tqdm(dataloaders['test']):
        inputs = batch['X'].to(device)
        labels = batch['y'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        test_running_loss += loss.item() * inputs.size(0)
        test_running_corrects += torch.sum(preds == labels.data)
        test_dataset_size += inputs.size(0)

        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

    test_loss = test_running_loss / float(test_dataset_size)
    test_acc = test_running_corrects.float() / float(test_dataset_size)

    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_report = classification_report(test_labels, test_preds)

    print('Test Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Balanced Acc: {:.4f}'.format(
        test_loss, test_acc, test_f1, test_balanced_acc))
    print(test_report)

    # Plot confusion matrix for test set
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Test Set')
    plt.show()

    return model
