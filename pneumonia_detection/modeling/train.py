import torch
from tqdm import tqdm
from pathlib import Path
from ..config import MODEL_DIR, NUM_EPOCHS, LEARNING_RATE

def train_model(model, train_loader, val_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0
    MODEL_DIR.mkdir(exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {val_acc:.3f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            print('Saving model...')
            torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
            best_val_acc = val_acc