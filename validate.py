import argparse
import logging

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform

from timm.models import create_model, load_checkpoint

# logger setting
_logger = logging.getLogger("train")

# configs setting
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda", type=str, help='training device')
parser.add_argument("--batch-size", default=128, type=int, help='batch size')
parser.add_argument("--model", default="resnet50", type=str, help='model name')
parser.add_argument("--model-weight", default='./weights/resnet/resnet50_cifar10.pth.tar', type=str,
                    help='weight path of model')
parser.add_argument("--num-classes", default=10, type=int, help='number of classes of the dataset')
parser.add_argument("--amp", action='store_true', default=False)


# main function
def main():
    # process args and some paths
    args = parser.parse_args()
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform setting
    transforms = transform.Compose([
        transform.Resize(224),
        transform.ToTensor(),
    ])

    # create dataset
    eval_dataset = datasets.CIFAR10(root="datasets", train=False, transform=transforms, download=False)

    # create dataloader
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    amp_autocast = torch.cuda.amp.autocast
    # create model
    model = create_model(args.model, pretrained=False, num_classes=args.num_classes)
    load_checkpoint(model, args.model_weight)
    model.cuda()

    # validation
    try:
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for batch_idx, (image, labels) in enumerate(eval_loader):
                image = image.cuda()
                labels = labels.cuda()
                with amp_autocast():
                    preds = model(image)
                pred_labels = torch.argmax(preds, dim=1)
                correct += (pred_labels == labels).sum()
                total += pred_labels.shape[0]
                acc = 100.0 * correct / total
    except KeyboardInterrupt:
        pass
    if acc is not None:
        print(f'***** Accuracy of the model is {acc:.2f} *****.')


if __name__ == "__main__":
    main()
