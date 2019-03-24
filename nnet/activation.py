
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors

    Args:
        z (torch.tensor): values to run sigmoid on. Size(batch_size, N_hn)
    
    Returns:
        result (torch.tensor): values after running through sigmoid. Size(batch_size, N_hn)
    """
    result = torch.div(torch.ones(z.size()), torch.add(torch.ones(z.size()), torch.exp(-1*z)))
    return result

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    Args:
        z (torch.tensor): values to run derivative of sigmoid on. Size(batch_size, N_hn)
    
    Returns:
        grad_sigmoid (torch.tensor): values after running through derivative of sigmoid. Size(batch_size, N_hn)
    """
    grad_sigmoid = torch.mul(sigmoid(z), torch.add(torch.ones(z.size()), -1, sigmoid(z)))
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    Args:
        x (torch.tensor): values to run softmax on. Size(batch_size, N_out)
    
    Returns:
        stable_softmax (torch.tensor): values after running through softmax. Size(batch_size, N_out)
    """
    shifted_x = torch.add(x, -1, torch.max(x , 1)[0].unsqueeze(1))
    exp_shifted_x = torch.exp(shifted_x)
    stable_softmax = torch.div(exp_shifted_x, torch.sum(exp_shifted_x, dim = 1).unsqueeze(1))
    return stable_softmax

if __name__ == "__main__":
    pass