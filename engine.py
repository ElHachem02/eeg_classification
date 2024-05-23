import sys
import time
import copy
import torch
import torch.nn as nn
import tqdm
from timm.utils import NativeScaler
from model import EEGTransformer
from timm.optim import AdamP
from timm.scheduler import CosineLRScheduler



def prepare_training():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = EEGTransformer().to(device)
    
    # Define optimizer parameters directly
    optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Define scheduler parameters directly
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=50,  # total number of epochs
        lr_min=1e-6,
        warmup_t=5,  # warmup epochs
        warmup_lr_init=1e-4,
    )

    criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    print(device)
    return model, optimizer, lr_scheduler, criterion, device, loss_scaler

def train_model(model, criterion, optimizer, scheduler, device, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(50):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch+1, 50))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            
            for batch in tqdm.tqdm(dataloaders[phase]):
                inputs = batch['X'] 
                labels = batch['y']

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

            if phase == 'train':
                scheduler.step(epoch=epoch)
                
            # Ensure dataset_size is a float32 tensor
            dataset_size = torch.tensor(dataset_size, dtype=torch.float32, device=running_corrects.device)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.float() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model