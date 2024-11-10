import torch
from torchvision.ops import box_iou

from cc import cc


def evaluate_model(model, test_loader, device):
    print(cc("GREEN", "Beginning evaluation..."))
    model.eval()

    total_iou = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Ground-truth
                gt_boxes = targets[i]["boxes"].to(device)
                gt_labels = targets[i]["labels"].to(device)

                # Predictions
                pred_boxes = output["boxes"]
                pred_labels = output["labels"]

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_iou_per_pred = ious.max(dim=1)[0]
                    avg_iou = max_iou_per_pred.mean().item()
                    total_iou += avg_iou
                    total_images += 1

                    print(f"Image {i + 1}: Avg IoU = {avg_iou:.4f}")
                else:
                    print(f"Image {i + 1}: No detections or ground-truth boxes to compare.")

    if total_images > 0:
        mean_iou = total_iou / total_images
        print(f"\nMean IoU over the test set: {mean_iou:.4f}")
    else:
        print("No valid predictions to calculate IoU.")
