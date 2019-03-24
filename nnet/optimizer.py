
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent

    Args:
        weights (dict of torch.tensor): weights b/w the layers of the neural network in a dictionary. Size(3 = no. of hidden layers)
        biases (dict of torch.tensor): biases b/w the layers of the neural network. Size(3 = no. of hidden layers)
        dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
        db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
        dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
        db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
        dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
        db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        lr (float): learning rate.
    Returns:
        nweights (dict of torch.tensor): dictionary of loss gradients w.r.t weights. Size like weights
        bweights (dict of torch.tensor): dictionary of loss gradients w.r.t biases. Size like biases
    """
    nweights = {}
    nbiases = {}
    nweights['w1'] = torch.add(weights['w1'], -lr, dw1)
    nweights['w2'] = torch.add(weights['w2'], -lr, dw2)
    nweights['w3'] = torch.add(weights['w3'], -lr, dw3)
    nbiases['b1'] = torch.add(biases['b1'], -lr, db1)
    nbiases['b2'] = torch.add(biases['b2'], -lr, db2)
    nbiases['b3'] = torch.add(biases['b3'], -lr, db3)


    return nweights, nbiases

if __name__ == "__main__":
    pass
