
import os
import torchvision.datasets
import torch.utils.data as data
from torchvision import transforms


class Load:
    def load_data(self, path, train_size, valid_size, nw):
        data_transform = {"train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                          "valid": transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

        flag = True
        if os.path.exists(path):
            flag = False

        train_datasets = torchvision.datasets.CIFAR10(root=path, train=True, download=flag, transform=data_transform["train"])
        train_loader = data.DataLoader(dataset=train_datasets, batch_size=train_size, shuffle=True, num_workers=nw)

        valid_datasets = torchvision.datasets.CIFAR10(root=path, train=False, download=flag, transform=data_transform["valid"])
        valid_loader = data.DataLoader(dataset=valid_datasets, batch_size=valid_size, shuffle=True, num_workers=nw)

        return train_loader, valid_loader
