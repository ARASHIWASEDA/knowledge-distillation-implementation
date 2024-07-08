import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataset(dataset_name, download, batch_size, num_workers):
    if dataset_name == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        eval_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root="datasets", train=True, transform=train_transforms,
                                         download=download)
        eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=eval_transforms,
                                        download=download)
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        eval_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root="datasets", train=True, transform=train_transforms,
                                          download=download)
        eval_dataset = datasets.CIFAR100(root="datasets", train=False, transform=eval_transforms,
                                         download=download)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader
