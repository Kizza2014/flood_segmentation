import torch
from .dice_score import dice_loss
import torch.functional as F

def evaluate_model(model, dataloader, device, amp):
    model.eval()
    dice_scores = []
    with torch.no_grad():  # Không cần tính gradient khi đánh giá
        for batch in dataloader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)

            # Tính Dice Score
            pred = F.softmax(masks_pred, dim=1)
            dice = dice_loss(pred, true_masks, multiclass=True)

            dice_scores.append(dice.item())

    # Trả về Dice Score trung bình
    return sum(dice_scores) / len(dice_scores)