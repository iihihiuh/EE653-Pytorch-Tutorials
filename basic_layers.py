import torch
import torch.nn as nn
import torch.optim as optim

def main(args):
    # a Linear layer + ReLU
    # https://github.com/pytorch/pytorch/blob/33b9726e6b3b11902f62b85f1e34fe0599ae17f5/torch/nn/modules/linear.py#L39
    # __init_ method is called
    Conv = nn.Linear(1024, 2048)
    ReLU = nn.ReLU(inplace=False)

    # an optimizer to update parameters
    opt = optim.SGD(Conv.parameters(), lr=0.01)
    

    # check if gradients are enabled
    print(Conv.weight.requires_grad, Conv.bias.requires_grad)

    # get a random input
    x = torch.randn(3, 1024)
    # get a random gradients
    grad = torch.randn(3, 2048)

    # forward pass
    # forward method is called
    y = ReLU(Conv(x))
    print(y.size())
    # backward pass
    y.backward(gradient=grad)

    # check gradients
    print(Conv.weight.grad.size(), Conv.bias.size())
    # checkgradients
    opt.step()
    # clear gradients after step
    opt.zero_grad()
    print("\nGradients become zero after zero_grad")
    print(Conv.weight.grad.sum(), Conv.bias.grad.sum())
    return 0


if __name__ == "__main__":
    main(None)