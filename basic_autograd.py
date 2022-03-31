import torch


def main(args):
    # pytorch autograd automatically keeps tracks of the computational graphs
    # and gradients associated with it

    # For example, a linear layer
    # x @ w + bias
    Weights = torch.randn(1024, 2048)
    Bias = torch.randn(2048) 
    print(Weights.requires_grad, Bias.requires_grad)   

    # set to true if you need their gradients
    Weights.requires_grad = True
    Bias.requires_grad = True
    
    # get a random input
    x = torch.randn(10, 1024)

    # get a random gradient
    grads = torch.randn(10, 2048)

    # forward pass
    y = x @ Weights + Bias

    # backward pass
    # loss.backward()
    y.backward(gradient=grads)

    print(Weights.grad.size(), Bias.grad.size())
    
    # update Parameters
    Weights = 0.1 * Weights.grad - Weights
    Bias = 0.1 * Bias.grad - Bias

    return 0


if __name__ == "__main__":
    main(None)