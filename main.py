from data import generate_data_LOSO
from engine import train_model, prepare_training
import torch

if __name__ == "__main__":
    val_sub = "dimi"
    test_sub = "ronan"
    dataloaders = generate_data_LOSO(val_sub, test_sub, batch_size=1, num_workers=0)
    
    device="mps"
    
    print(f"shape is {len(dataloaders)}")
    
    # Extract training labels from the dataloader
    train_y = []
    for batch in dataloaders['train']:
        train_y.extend(batch['y'].cpu().numpy())
    
    model, optimizer, lr_scheduler, criterion, device, loss_scaler = prepare_training(train_y)
    trained_model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders)

    # Save the trained model
    torch.save(trained_model.state_dict(), "eeg_transformer_model.pth")