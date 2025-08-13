import torch
from pneumonia_detection.dataset import get_dataloaders
from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.trainer import run_training
from pneumonia_detection.config import MODEL_DIR

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = PneumoniaCNN().to(device)

    save_path = (MODEL_DIR / "best_cnn_model.pth").as_posix()
    run_training(model, train_loader, val_loader, device, save_path=save_path)

    print('Training completed! Best model saved at:', save_path)

if __name__ == '__main__':
    main()
