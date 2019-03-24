
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    Args:
        outputs (torch.tensor):outputs predicted by neural network. Size (batch_size, N_out)
        labels (torch.tensor): correct labels. Size (batch_size)
    
    Returns:
        creloss (float): average cross entropy loss
    """
    EPS = 1e-12
    label_expanded = torch.zeros(outputs.size())
    for index, label in enumerate(labels):
        label_expanded[index][label] = 1
    true_loss = torch.mul(label_expanded, torch.log(torch.add(outputs, EPS)))
    loss = true_loss #torch.add(true_loss, false_loss)
    creloss = -1*torch.sum(loss)/len(labels)
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z).

    Args:
        outputs (torch.tensor):outputs predicted by neural network. Size (batch_size, N_out)
        labels (torch.tensor): correct labels. Size (batch_size)
    
    Returns:
        avg_grads (torch.tensor): derivate of cross entropy loss w.r.t weighted sum of inputs, 
        found using chain rule of differentiation. Size (batch_size, N_out)
    """
    ones = torch.ones(outputs.size())
    label_expanded = torch.zeros(outputs.size())
    for index, label in enumerate(labels):
        label_expanded[index][label] = 1
    true_grads = torch.add(outputs, -1, label_expanded)

    avg_grads = torch.mul(true_grads,1/outputs.size()[0])
    return avg_grads

if __name__ == "__main__":
    pass
