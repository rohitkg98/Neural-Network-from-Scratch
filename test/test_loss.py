from context import nnet
from nnet import loss, activation

import unittest
import torch
from torch import nn
import math
import numpy as np

class TestLossModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests
    def test_cross_entropy(self):
        # settings
        batch_size = 4
        N_out = 10
        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        outputs_s = activation.softmax(outputs)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)

        closs = nn.CrossEntropyLoss()
        creloss = loss.cross_entropy_loss(outputs_s, labels)
        assert type(creloss) == float
        # write more robust and rigourous test cases here
        self.assertTrue(math.isclose(creloss, closs(outputs,labels).item(), rel_tol=0.0009))

    def test_delta_cross_entropy_loss(self):
        # settings
        batch_size = 4
        N_out = 10
        precision = 0.000001
        
        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float, requires_grad = True)
        labels = torch.randint(high=10, size=(batch_size,), dtype=torch.long)
        outputs_s = outputs
         
        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)

        closs = nn.CrossEntropyLoss()
        xloss = closs(outputs, labels)
        xloss.backward()
        

        grads_creloss = loss.delta_cross_entropy_softmax(outputs, labels)

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        # write more robust test cases here
        # you should write gradient checking code here

if __name__ == '__main__':
    unittest.main()