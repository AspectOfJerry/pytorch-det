import torch

from cc import cc


def get_device(cuda_avail):
    device = torch.device("cuda" if cuda_avail else "cpu")

    print(cc("BLUE", f"CUDA available: {cuda_avail}"))
    print(cc("BLUE", f"Using device: {device}"))

    if device == torch.device("cuda"):
        print(cc("BLUE", f"Number of GPUs: {torch.cuda.device_count()}"))
        print(cc("BLUE", f"GPU 0: {torch.cuda.get_device_name(0)}"))

    return device
