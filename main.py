from data import generate_data_LOSO, generate_data_LOPO
from engine import train_model, prepare_training
import torch

if __name__ == "__main__":
    val_subject = "ronan"
    test_subject = "lea"
    device="mps"

    data_loaders_LOPO = generate_data_LOPO()

    data_loaders_LOSO = generate_data_LOSO(val_subject, test_subject)
    print(f"shape is {len(data_loaders_LOSO)}")
    
    model, optimizer, lr_scheduler, criterion, device, loss_scaler = prepare_training()
    trained_model = train_model(model, criterion, optimizer, lr_scheduler, device, data_loaders_LOSO)

    # Save the trained model
    torch.save(trained_model.state_dict(), "eeg_transformer_model.pth")