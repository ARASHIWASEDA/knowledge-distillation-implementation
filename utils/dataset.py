import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from timm.data import ImageDataset
from torchvision.datasets import CIFAR100
from PIL import Image


class ImageNetInstanceSample(ImageDataset):
    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_dataset(args):
    if args.dataset == 'cifar10':
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
                                         download=args.dataset_download)
        eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=eval_transforms,
                                        download=args.dataset_download)
    elif args.dataset == 'cifar100':
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
        if args.distiller == 'crd':
            train_dataset = CIFAR100InstanceSample(root="datasets", train=True, is_sample=True, k=args.crd_k,
                                                   transform=train_transforms)
        else:
            train_dataset = datasets.CIFAR100(root="datasets", train=True, transform=train_transforms,
                                              download=args.dataset_download)
        eval_dataset = datasets.CIFAR100(root="datasets", train=False, transform=eval_transforms,
                                         download=args.dataset_download)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    return train_dataset, eval_dataset, train_loader, eval_loader
