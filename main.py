import math
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from cc import cc
from modules.dataset import new_datasets, new_data_loaders
from modules.evaluate import evaluate_model
from modules.model import new_model, new_optimizer, new_scheduler
from modules.train import train_epoch
from modules.utils import get_device

"""
Run configuration
"""
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"  # The directory will be created if it does not exist
model_save_path = os.path.join(OUTPUT_DIR, "inference_graph.pth")  # file name

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tensorboard_logs"))

"""
Training parameters
"""
# Total training cycles
NUM_EPOCHS = 36

# Number of images per batch
BATCH_SIZE = 32

# Initial step size, usually between 0.0001 and 0.01 ?
LEARNING_RATE = 0.0003

# Interval (in epochs) to decay the learning rate, usually 20-30% of the total number of epochs ?
STEP_SIZE = 6

# Factor by which the learning rate decays, usually around 0.5 ?
GAMMA = 0.670

# Number of classes in the dataset (excluding background)
NUM_CLASSES = 1

"""
Creating the model
"""
print(cc("YELLOW", "Initializing model..."))
model = new_model(out_features=NUM_CLASSES + 1)  # add 1 for the background class
model.train()

# Model summary
print(cc("GRAY", "Model summary:"))
print(cc("GRAY", str(summary(
    model,
    input_size=(BATCH_SIZE, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))))

# Logging training parameters
print(cc("CYAN", f"Number of epochs: {NUM_EPOCHS}"))
print(cc("CYAN", f"Batch size: {BATCH_SIZE}"))
print(cc("CYAN", f"Learning rate: {LEARNING_RATE}"))
print(cc("CYAN", f"Step size: {STEP_SIZE}"))
print(cc("CYAN", f"Gamma: {GAMMA}"))
print(cc("GRAY", "-------------------------"))

"""
Configuring devices
"""
print(cc("YELLOW", "Configuring devices..."))
DEVICE = get_device(torch.cuda.is_available())
print(cc("GRAY", "-------------------------"))

# Move model to configured device
model.to(DEVICE)

"""
Data preparation
"""
# Datasets
print(cc("YELLOW", "Creating datasets..."))
# Note: transforms are applied in the new_datasets function
train_dataset, test_dataset = new_datasets(data_dir=DATA_DIR, device=DEVICE)

# Data loaders
print(cc("YELLOW", "Creating data loaders..."))
cpu_count = 0
train_loader, test_loader = new_data_loaders(batch_size=BATCH_SIZE, train_dataset=train_dataset, test_dataset=test_dataset)

# Additional training details
batches_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = math.ceil(len(train_dataset) / BATCH_SIZE) * NUM_EPOCHS

print(cc("CYAN", f"Training dataset: {len(train_dataset)} images"))
print(cc("CYAN", f"Batches per epoch: {batches_per_epoch}"))
print(cc("CYAN", f"Total training batches: {total_steps}"))
print(cc("CYAN", f"Validation dataset: {len(test_dataset)} images"))
print(cc("GRAY", "-------------------------"))

optimizer = new_optimizer(model, LEARNING_RATE)
scheduler = new_scheduler(optimizer, STEP_SIZE, GAMMA)

input(cc("GREEN", "Ready to begin training with the current configuration. Press any key to continue . . ."))

start_time = time.time()

# Training loop
total_steps = NUM_EPOCHS * len(train_loader)
# prev_loss, prev_lr = 0, optimizer.param_groups[0]["lr"]

for epoch in range(NUM_EPOCHS):
    train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        writer=writer,
        epoch_count=epoch,
        total_steps=total_steps
    )
    # Note: the scheduler step is called in the train_epoch function

print(cc("GREEN", f"Training complete! Took {time.time() - start_time:.3f} seconds"))

"""
Post-training and cleanup
"""
# Close the TensorBoard SummaryWriter
writer.close()

# Save .pth model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
print(cc("GRAY", f"Trained model saved at {model_save_path}"))

"""
Evaluation
"""
input("Press any key to proceed to evaluation . . .")

# Run evaluation on the test dataset
evaluate_model(model=model, test_loader=test_loader, device=DEVICE)

"""
print("Conversion to TFLite does not work at the moment!")
input("Press any key to continue (exporting to ONNX then to TFLite) . . .")

# onnx~=1.14.1
# tf2onnx~=1.15.1

# Export the PyTorch model to ONNX format
input_shape = (BATCH_SIZE, 3, 3024, 3024)
dummy_input = torch.randn(input_shape)
onnx_path = os.path.join(OUTPUT_DIR, "saved_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=12)

# Convert ONNX model to TFLite
onnx_model = onnx.load(onnx_path)

# Specify target_opset and optimize
tflite_model = onnx_tf.backend.prepare(onnx_model, strict=False)
# https://github.com/onnx/onnx-tensorflow/issues/763
# optimized_onnx_model = tflite_model.graph.as_graph_def()
# tflite_optimized_model = tf2onnx.convert.from_graph_def(optimized_onnx_model, opset=12, output_path=None)

tflite_path = os.path.join(OUTPUT_DIR, "saved_model.tflite")
tflite_model.export_graph(tflite_path)
# with open(tflite_path, "wb") as f:
#     f.write(tflite_model)
"""
