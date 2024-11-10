import torch
import torchvision.transforms as transforms

from custom_dataset import CustomDataset


def new_datasets(data_dir, device):
    data_transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
    data_transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CustomDataset(data_dir, "train", transform=data_transform_train, device=device)
    test_dataset = CustomDataset(data_dir, "test", transform=data_transform_test, device=device)

    return train_dataset, test_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def new_data_loaders(batch_size, train_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return train_loader, test_loader
