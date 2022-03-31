import torch
import torch.nn as nn
from vgg import VGG


def main(args):
    # model & inputs
    # default location is on CPUs
    model = VGG()
    x = torch.randn(3, 3, 224, 224)

    # using to send models & inputs to GPUs
    model = model.to("cuda:0")
    x = x.to("cuda:0")
    x_gpu2 = x.to("cuda:1")

    y_pred = model(x)

    # y_pred2 = model(x_gpu2)

    # send y_pred back to cpu
    y_pred = y_pred.to("cpu:0")
    return 0


if __name__ == "__main__":
    main(None)