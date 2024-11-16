import time

from cc import cc, ccnum

# Tracking variables
prev_lr = 0
next_lr = 0
prev_loss = 0
next_loss = 0


def _train_step(model, images, targets, optimizer, device):
    model.train()  # Ensure the model is in training mode
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Forward pass
    losses = model(images, targets)
    total_loss = sum(loss for loss in losses.values())

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
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
