import torch
import torchvision

from custom_dataset import CustomDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def new_datasets(data_dir, device, data_transform_train=None, data_transform_test=None):
    train_dataset = CustomDataset(data_dir, "train", transform=data_transform_train, device=device)
    test_dataset = CustomDataset(data_dir, "test", transform=data_transform_test, device=device)

    return train_dataset, test_dataset


def new_data_loaders(batch_size, train_dataset, test_dataset, cpu_count=0):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count, collate_fn=collate_fn)
    return train_loader, test_loader
