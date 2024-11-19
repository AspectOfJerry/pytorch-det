import time
from collections import OrderedDict

import torch

from cc import cc, ccnum

# Tracking variables
prev_lr = 0
next_lr = 0
prev_loss = 0
next_loss = 0

"""
fasterrcnn_mobilenet_v3_large_320_fpn losses
- loss_objectness: Measures how well the RPN predicts whether a region contains an object.
- loss_rpn_box_reg: Measures how well the RPN refines candidate box coordinates.
- loss_classifier: Measures the accuracy of the predicted class labels for detected objects.
- loss_box_reg: Measures the accuracy of bounding box predictions for detected objects.
"""


def _train_step(model, images, targets, optimizer, device, clip_threshold=100.0):
    model.train()  # Ensure the model is in training mode
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Use the model transforms to preprocess images and targets
    images, targets = model.transform(images, targets)

    # Forward pass
    # Extract features from the backbone
    feature_maps = model.backbone(images.tensors)

    # Ensure feature_maps is always an OrderedDict even if backbone returns single tensor
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = OrderedDict([("0", feature_maps)])

    # Generate region proposals using the Region Proposal Network (RPN)
    proposals, rpn_losses = model.rpn(images, feature_maps, targets)

    # Perform Region of Interest (ROI) pooling
    detections, roi_losses = model.roi_heads(feature_maps, proposals, images, targets)

    losses = {}
    losses.update(rpn_losses)
    losses.update(roi_losses)

    # losses = model(images, targets)
    total_loss = sum(loss for loss in losses.values())

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Clip parameters to avoid exploding gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            max_grad = grad.max()
            min_grad = grad.min()

            if max_grad > clip_threshold or min_grad < -clip_threshold:
                print(cc("RED", f"Clipping gradient in parameter '{name}' (max={max_grad.item()}, min={min_grad.item()})"))
                grad.data.clamp_(-clip_threshold, clip_threshold)

    optimizer.step()

    return total_loss.item(), losses


def train_epoch(model, train_loader, optimizer, scheduler, num_epochs, device, writer, epoch_count, total_steps):
    global prev_loss, next_loss, prev_lr, next_lr
    model.train()
    epoch_timer = time.time()
    print(cc("GREEN", f"Beginning epoch {epoch_count + 1}/{num_epochs}..."))

    for step, (images, targets) in enumerate(train_loader):
        # Call train_step for a single batch
        total_loss, losses = _train_step(model, images, targets, optimizer, device)

        # Log to TensorBoard
        step_count = epoch_count * len(train_loader) + step
        writer.add_scalar("Loss/total_loss", total_loss, step_count)
        for i, lr in enumerate(scheduler.get_last_lr()):
            writer.add_scalar(f"Learning Rate/param_group{i}", lr, step_count)

        # Console logging
        print(cc("BLUE", f"Epoch [{epoch_count + 1}/{num_epochs}] - Step {step_count + 1}/{total_steps}:"))  # count starts from 0
        print(cc("CYAN", f"Total loss: {total_loss}" + cc("GRAY", f" ({next_loss:.2e})")))
        for i, lr in enumerate(scheduler.get_last_lr()):
            print(cc("CYAN", f"Learning rate: {lr}" + cc("GRAY", f" ({lr:.2e})")))
        print(cc("BLUE",
                 f"- Losses:\n"
                 + "\n".join([f"  - {key}: {loss}" for key, loss in losses.items()]) + "\n"))

        # Training loss difference
        next_loss = total_loss
        delta_loss = next_loss - prev_loss
        print(cc("CYAN", f"Training loss delta: {ccnum(delta_loss, reverse=True)}" + cc("GRAY", f" ({delta_loss:.2e})")))
        prev_loss = next_loss

        # Learning rate difference
        next_lr = scheduler.get_last_lr()[0]
        delta_lr = next_lr - prev_lr
        print(cc("CYAN", f"Learning rate delta: {ccnum(delta_lr, reverse=True)}" + cc("GRAY", f" ({delta_lr:.2e})")))
        prev_lr = next_lr

    print(cc("GREEN", f"Epoch [{epoch_count + 1}/{num_epochs}] complete in {time.time() - epoch_timer:.3f} seconds"))  # count starts from 0
    scheduler.step()  # Step the scheduler after each epoch
