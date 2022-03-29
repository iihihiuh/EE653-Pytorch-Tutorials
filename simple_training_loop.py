import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from vgg import VGG
from tqdm import tqdm

def dataLoader(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    data_set = datasets.CIFAR10(root='~/', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)

    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True)
    return train_loader


def main(args):
    model = VGG().to("cuda:0")
    batch_size = 20

    dataset = dataLoader(batch_size)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss().to("cuda:0")

    for image, target in tqdm(dataset):
        image = image.to("cuda:0")
        target = target.to("cuda:0")

        loss = loss_fn(model(image), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return 0


if __name__ == "__main__":
    main(None)