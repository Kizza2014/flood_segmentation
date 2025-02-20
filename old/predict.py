from unet import UNet
import torch


checkpoint = torch.load('./checkpoints/checkpoint_epoch100.pth')

model = UNet(n_channels=3, n_classes=2)
model.load_state_dict(checkpoint)

model.eval()