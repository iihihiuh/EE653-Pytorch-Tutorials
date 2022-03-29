import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # a vector descriping VGG16 dimensions
        vec = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]

        self.listModule = []
        in_channels = 3

        for v in vec:
            if v == "M":
                self.listModule += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                self.listModule += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.features = nn.Sequential(*self.listModule)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.cl = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        ]
        self.classifier = nn.Sequential(*self.cl)

    
    def forward(self, x):
        # x = self.features(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


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