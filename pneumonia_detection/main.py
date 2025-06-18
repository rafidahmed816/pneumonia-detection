import torch
from .dataset import get_dataloaders
from .modeling.model import PneumoniaModel
from .modeling.train import train_model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # Initialize model
    model = PneumoniaModel().to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    print('Training completed!')

if __name__ == '__main__':
    main()