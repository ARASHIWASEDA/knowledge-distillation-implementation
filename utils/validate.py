import torch


def validate(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images.cuda()
        labels.cuda()
        logits = model(images)
        with torch.no_grad():
            logits = torch.argmax(logits, dim=1)
            correct += (logits == labels).sum()
            total += labels.shape[0]
    return 100.0 * correct / total
