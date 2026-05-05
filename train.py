import torch
import torch.optim as optim
from sod_model import SODModel
from data_loader import get_loaders

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SODModel().to(device)
    train_loader, val_loader, _ = get_loaders()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss() # Simplified for script

    for epoch in range(10):
        model.train()
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Dummy targets for script structure
            targets = torch.zeros_like(outputs).to(device) 
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

if __name__ == "__main__":
    train_model()
