# PyTorch object detection setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.11    |

### Dataset structure

The dataset should be structured as follows:

```
dataset
├── discarded (rejected images, should not happen)
├── annotations
│   ├── test
│   └── train
└── images
    ├── test
    └── train
```

### Run tensorboard (optional)

TensorBoard can be used to monitor the training process, including losses and metrics. To start TensorBoard, run:

```bash
tensorboard --logdir=./output/tensorboard_logs
```

This will display training progress in your browser.

### Using a venv (Recommended)

Creating a virtual environment helps manage project dependencies independently of other Python projects.
If you use PyCharm, its virtual environment manager makes this very easy and straightforward.

Alternatively, to set it up manually:

```bash
python -m venv venv
```

Activate the virtual environment before proceeding with the installation of dependencies. You might need to reopen a new terminal.

## Installing dependencies

From [PyTorch's website](https://pytorch.org/get-started/locally/), here are the installation commands:
Common dependencies will be installed along with PyTorch.

### CPU-only

Use this command to install PyTorch with only CPU support:

```bash
pip install torch torchvision torchaudio -r requirements.txt
```

### GPU (NVIDIA CUDA)

Use this command to install PyTorch with GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
```

> **Note**: PyTorch’s GPU version comes bundled with all necessary binaries, such as CUDA and cuDNN, so you don’t need to install them separately.

## Converting the .pth model to .tflite
