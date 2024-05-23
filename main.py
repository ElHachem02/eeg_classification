from data import generate_data_with_loaders
from engine import train_model, prepare_training
import torch

if __name__ == "__main__":
    data_dict, combined_data, dataloaders_dict, combined_dataloaders = generate_data_with_loaders()
    device="mps"
    
    print(f"shape is {len(combined_dataloaders)}")
    
    model, optimizer, lr_scheduler, criterion, device, loss_scaler = prepare_training()
    trained_model = train_model(model, criterion, optimizer, lr_scheduler, device, combined_dataloaders)

    # Save the trained model
    torch.save(trained_model.state_dict(), "eeg_transformer_model.pth")