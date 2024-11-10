# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |

## Running the code

To start training the model, run the `main.py` file (`python main.py`).

- Configure the training in the file under `Training parameters`

To test the model, run the `inference.py` file (`python inference.py`).

- Configure the inference in the file (specify the path to the image you want to test)

### Dataset structure

The dataset should be structured as follows:

```
dataset
├── empty (empty directory to be filled with rejected images with no annotations, should not happen)
├── annotations
│   ├── test
│   └── train
└── images
    ├── test
    └── train
```

### Run tensorboard

To run tensorboard, use the following command:

```bash
tensorboard --logdir=./output/tensorboard_logs
```

### venv

To run the project in a Python virtual environment, run the following commands:

```bash
python -m venv venv
```

## CPU-only

Install the dependencies for CPU-only training using the following command:

From PyTorch's website:

```bash
pip install torch torchvision torchaudio
```

and to install common dependencies for both CPU and CUDA training:

```bash
pip install -r requirements.txt
```

```
torchinfo~=1.8.0
opencv-python~=4.10.0.84
pillow~=10.2.0
tensorboard~=2.18.0
```

## GPU (NVIDIA CUDA)

From PyTorch's website:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

and to install common dependencies for both CPU and CUDA training:

```bash
pip install -r requirements.txt
```

```
torchinfo~=1.8.0
opencv-python~=4.10.0.84
pillow~=10.3.0
tensorboard~=2.18.0
```

If you have an NVIDIA GPU, you can install the CUDA toolkit and cuDNN to enable GPU support. If you don't have an NVIDIA GPU, skip the following steps.

- Download and install [CUDA 12.4 for Windows](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- Download and install [Visual Studio](https://visualstudio.microsoft.com/) with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v... - VS 20.. C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (any compatible version)
    - Create an NVIDIA developer account if you don't have one
- Extract the cuDNN zip file
    - Copy the files from the `bin` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
    - Copy the files from the `include` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include`
    - Copy the files from the `lib\x64` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64`
