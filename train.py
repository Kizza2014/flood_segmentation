import torch
import torchvision.transforms as transforms
from dataset import FloodAreaDataset
from torch.utils.data import random_split, DataLoader
from res_unet.model import ResNetUNet
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_data(image_dir, mask_dir, batch_size=8, patch_size=(512, 512), seed=43):
    torch.manual_seed(seed)

    image_transform = transforms.Compose([
    transforms.Resize(size=patch_size, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=patch_size, antialias=False)
    ])

    train_augmentations = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ])

    dataset = FloodAreaDataset(image_dir, mask_dir, image_transform=image_transform, mask_transform=mask_transform)

    # Define the sizes for each split
    dataset_size = len(dataset)
    test_size = int(0.15 * dataset_size)
    val_size  = int(0.15 * dataset_size)
    train_size = dataset_size - test_size - val_size

    # Use random_split to create train, test, and val datasets
    train_dataset, temp_dataset = random_split(dataset, [train_size, test_size + val_size], generator=torch.Generator().manual_seed(42))
    test_dataset, val_dataset = random_split(temp_dataset, [test_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset.augmentations = train_augmentations

    # Create DataLoader instances for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader
    }

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet_unet = ResNetUNet(in_channels=3, out_channels=2, resnet_type="resnet34").to(device)

    # Freeze parameters in blocks 1, 2, 3, and 4
    for block in [resnet_unet.block1, resnet_unet.block2, resnet_unet.block3, resnet_unet.block4]:
        for param in block.parameters():
            param.requires_grad_(False)

    return resnet_unet

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, early_stop_patience=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for _, (images, masks) in tqdm(enumerate(train_loader), desc='Training', total=len(train_loader), unit='batch'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation', total=len(val_loader), unit='batch'):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print and check for early stopping
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping after {early_stop_patience} epochs without improvement.')
            break

    return train_losses, val_losses


if __name__ == "__main__":
    data = initialize_data('Image', 'Mask')
    model = initialize_model()

    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    LR_FACTOR = 0.5
    LR_PATIENCE = 2
    EARLY_STOP_PATIENCE = 4
    criterion = smp.losses.DiceLoss('multiclass')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, verbose=False)
    train_losses, val_losses = train_model(model, data['train'], data['val'], criterion, optimizer, scheduler, NUM_EPOCHS, EARLY_STOP_PATIENCE)

    plt.figure(figsize=(10, 5))
    sns.lineplot(train_losses, label='Train Loss', color='blue')
    sns.lineplot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_plot.png')