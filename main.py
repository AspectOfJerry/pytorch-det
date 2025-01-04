import math
import os
import time

import torch
import torchinfo
import torchvision
from torch.utils.tensorboard import SummaryWriter

import modules.dataset as datasets
import modules.evaluate as evaluate
import modules.model as models
import modules.train as trainer
import modules.utils as utils
from cc import cc

# Run configuration
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"  # The directory will be created if it does not exist
model_save_path = os.path.join(OUTPUT_DIR, "inference_graph.pth")  # file name

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tensorboard_logs"))

os.makedirs(os.path.join(DATA_DIR, "empty", "images"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "empty", "annotations"), exist_ok=True)

"""
Training parameters
- Number of epochs: Total training cycles
- Batch size: Number of images per batch
- Number of classes: Number of classes in the dataset (excluding background)
- Data transformation: Data augmentation and normalization for training
"""
# Total training cycles
NUM_EPOCHS = 36

# Number of images per batch
BATCH_SIZE = 32

# Number of classes in the dataset (excluding background)
NUM_CLASSES = 1

# Data augmentation and normalization for training
DATA_TRANSFORM_TRAIN = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Data transformation and normalization for testing
DATA_TRANSFORM_TEST = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

"""
Optimizer parameters
- Learning rate: Initial learning rate
- Betas: Coefficients used for computing running averages of gradient and its square
- Epsilon: Term added to the denominator to improve numerical stability
- Weight decay: L2 penalty
- AMSGrad: Whether to use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond"
https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
"""
# Initial learning rate. Default: 1e-3 (0.001)
LEARNING_RATE = 0.0005

# Coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.999)
BETAS = (0.9, 0.999)

# Term added to the denominator to improve numerical stability. Default: 1e-8 (0.00000001)
EPS = 0.00000001  # 1e-8

# Weight decay (L2 penalty). Default: 1e-2 (0.01)
WEIGHT_DECAY = 0.0005

# Use the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond". default: False
AMSGRAD = False

"""
Scheduler parameters
- Step size: Period of learning rate decay
- Gamma: Multiplicative factor of learning rate decay
- Last epoch: The index of the last epoch
"""
# Period of learning rate decay, usually 20-30% of the total number of epochs (?)
STEP_SIZE = 8

# Multiplicative factor of learning rate decay. Default: 0.1
GAMMA = 0.800

# The index of the last epoch. Default: -1
LAST_EPOCH = -1

# Create model
print(cc("YELLOW", f"Initializing model with {NUM_CLASSES} classes (excluding background)..."))
model = models.new_model(out_features=NUM_CLASSES + 1)  # add 1 for the background class

optimizer = models.new_optimizer(
    model=model, learning_rate=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD
)
scheduler = models.new_scheduler(
    optimizer=optimizer, step_size=STEP_SIZE, gamma=GAMMA, last_epoch=LAST_EPOCH
)

model.train()

# Model summary
print(cc("GRAY", "Model summary:"))
print(cc("GRAY", str(torchinfo.summary(
    model,
    input_size=(BATCH_SIZE, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))))

# Logging parameters
print(cc("BLUE", "Training parameters"))
print(cc("CYAN", f"Number of epochs: {NUM_EPOCHS}"))
print(cc("CYAN", f"Batch size: {BATCH_SIZE}"))
print(cc("GRAY", "-------------------------"))
print(cc("BLUE", "Optimizer parameters"))
print(cc("CYAN", f"Learning rate: {LEARNING_RATE}"))
print(cc("CYAN", f"Betas: {BETAS}"))
print(cc("CYAN", f"Epsilon: {EPS}"))
print(cc("CYAN", f"Weight decay: {WEIGHT_DECAY}"))
print(cc("CYAN", f"Use AMSGrad: {AMSGRAD}"))
print(cc("GRAY", "-------------------------"))
print(cc("BLUE", "Scheduler parameters"))
print(cc("CYAN", f"Step size: {STEP_SIZE}"))
print(cc("CYAN", f"Gamma: {GAMMA}"))
print(cc("CYAN", f"Last epoch: {LAST_EPOCH}"))
print(cc("GRAY", "-------------------------"))

# Configure device
print(cc("YELLOW", "Configuring devices..."))
DEVICE = utils.get_device(torch.cuda.is_available())
print(cc("GRAY", "-------------------------"))

# Move model to configured device
model.to(DEVICE)

# Datasets
print(cc("YELLOW", "Creating datasets..."))
# Note: transforms are applied in the new_datasets function
train_dataset, test_dataset = datasets.new_datasets(
    data_dir=DATA_DIR, device=DEVICE, data_transform_train=DATA_TRANSFORM_TRAIN, data_transform_test=DATA_TRANSFORM_TEST
)

# Data loaders
print(cc("YELLOW", "Creating data loaders..."))
train_loader, test_loader = datasets.new_data_loaders(
    batch_size=BATCH_SIZE, train_dataset=train_dataset, test_dataset=test_dataset, cpu_count=0
)

# Additional training details
batches_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
total_steps = math.ceil(len(train_dataset) / BATCH_SIZE) * NUM_EPOCHS

print(cc("CYAN", f"Training dataset: {len(train_dataset)} images"))
print(cc("CYAN", f"Batches per epoch: {batches_per_epoch}"))
print(cc("CYAN", f"Total training batches: {total_steps}"))
print(cc("CYAN", f"Validation dataset: {len(test_dataset)} images"))
print(cc("GRAY", "-------------------------"))

input(cc("GREEN", "Ready to begin training with the current configuration. Press any key to continue . . ."))

start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    trainer.train_epoch(
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

# Post training and cleanup
writer.close()  # close TensorBoard SummaryWriter

# Save .pth model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
print(cc("GRAY", f"Trained model saved at {model_save_path}"))

# Evaluation
input("Press any key to proceed to evaluation . . .")

# Run evaluation on the test dataset
evaluate.evaluate_model(model=model, test_loader=test_loader, device=DEVICE)

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
