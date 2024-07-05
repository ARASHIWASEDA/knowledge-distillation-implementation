import torchvision.datasets as datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader


def get_dataset(dataset_name, download, batch_size):
    if dataset_name == 'cifar10':
        train_transforms = transform.Compose([
            transform.Resize((256, 256)),
            transform.RandomCrop(224),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        eval_transforms = transform.Compose([
            transform.Resize((256, 256)),
            transform.CenterCrop(224),
            transform.ToTensor(),
            transform.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root="datasets", train=True, transform=train_transforms,
                                         download=download)
        eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=eval_transforms,
                                        download=download)
    else:
        train_transforms = transform.Compose([
            transform.Resize((256, 256)),
            transform.RandomCrop(224),
            transform.RandomHorizontalFlip(),
            transform.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        eval_transforms = transform.Compose([
            transform.Resize((256, 256)),
            transform.CenterCrop(224),
            transform.Normalize(mean=((0.5071, 0.4867, 0.4408)), std=(0.2675, 0.2565, 0.2761))
        ])
        train_dataset = datasets.CIFAR100(root="datasets", train=True, transform=train_transforms,
                                          download=download)
        eval_dataset = datasets.CIFAR100(root="datasets", train=False, transform=eval_transforms,
                                         download=download)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader
