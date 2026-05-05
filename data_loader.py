import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Using CIFAR-10 as proxy for SOD task
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split 70/15/15
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
