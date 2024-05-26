import sys
import time
import copy
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from timm.utils import NativeScaler
from model import EEGTransformer
from timm.optim import AdamP, AdamW
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

nb_epochs = 30

def prepare_training(train_y):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = EEGTransformer().to(device)
    
    # Define optimizer parameters directly
    optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Define scheduler parameters directly
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=nb_epochs,  # total number of epochs
        lr_min=1e-6,
        warmup_t=5,  # warmup epochs
        warmup_lr_init=1e-4,
    )

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Use class weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    loss_scaler = NativeScaler()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, loss_scaler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Lists to store metrics for plotting
    epochs_list = []
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    train_f1s, val_f1s, test_f1s = [], [], []
    train_balanced_accuracies, val_balanced_accuracies, test_balanced_accuracies = [], [], []

    for epoch in range(nb_epochs):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 50))
        print('-' * 10)

        # Each epoch has a training, validation, and test phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            
            all_labels = []
            all_preds = []

            for batch in tqdm.tqdm(dataloaders[phase]):
                inputs = batch['X'].to(device)
                labels = batch['y'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                dataset_size += inputs.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == 'train':
                scheduler.step(epoch=epoch)
                
            # Ensure dataset_size is a float32 tensor
            dataset_size = torch.tensor(dataset_size, dtype=torch.float32, device='cpu')

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.float() / dataset_size

            # Compute additional metrics
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            # Filter out predictions that are not in the true labels
            valid_classes = np.unique(all_labels_np)
            valid_preds_mask = np.isin(all_preds_np, valid_classes)
            all_preds_np = all_preds_np[valid_preds_mask]
            all_labels_np = all_labels_np[valid_preds_mask]

            f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=1)
            balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
            accuracy = accuracy_score(all_labels_np, all_preds_np)
            report = classification_report(all_labels_np, all_preds_np, zero_division=1)

            if phase == 'train':
                train_losses.append(epoch_loss.cpu().numpy())
                train_accuracies.append(epoch_acc.cpu().numpy())
                train_f1s.append(f1)
                train_balanced_accuracies.append(balanced_acc)
            elif phase == 'val':
                val_losses.append(epoch_loss.cpu().numpy())
                val_accuracies.append(epoch_acc.cpu().numpy())
                val_f1s.append(f1)
                val_balanced_accuracies.append(balanced_acc)
            else:  # phase == 'test'
                test_losses.append(epoch_loss.cpu().numpy())
                test_accuracies.append(epoch_acc.cpu().numpy())
                test_f1s.append(f1)
                test_balanced_accuracies.append(balanced_acc)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Balanced Acc: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, f1, balanced_acc, accuracy))
            print(report)

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epochs_list.append(epoch + 1)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot the metrics
    plot_metrics(epochs_list, train_losses, val_losses, test_losses,
                 train_accuracies, val_accuracies, test_accuracies,
                 train_f1s, val_f1s, test_f1s,
                 train_balanced_accuracies, val_balanced_accuracies, test_balanced_accuracies)

    return model

def plot_metrics(epochs, train_losses, val_losses, test_losses,
                 train_accuracies, val_accuracies, test_accuracies,
                 train_f1s, val_f1s, test_f1s,
                 train_balanced_accuracies, val_balanced_accuracies, test_balanced_accuracies):
    plt.figure(figsize=(20, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.plot(epochs, test_f1s, label='Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.legend()

    # Plot Balanced Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_balanced_accuracies, label='Train Balanced Accuracy')
    plt.plot(epochs, val_balanced_accuracies, label='Validation Balanced Accuracy')
    plt.plot(epochs, test_balanced_accuracies, label='Test Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy Over Epochs')
   





