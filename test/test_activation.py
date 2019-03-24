from context import nnet
from nnet import activation

import unittest
import torch
import math
import numpy as np

class TestActivationModule(unittest.TestCase):
    # Extra TODO: Write more rigorous tests

    def test_sigmoid(self):
        # This is an example test case. We encourage you to write more such test cases.
        # You can test your code unit-wise (functions, classes, etc.) 
        x = torch.FloatTensor([[-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50]])
        y = torch.FloatTensor([[4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999], 
                                [4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999]])
        precision = 0.0009
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(x) - y), precision).all())

    def test_delta_sigmoid(self):
        batch_size = 2
        N_hn = 8

        #x = torch.rand((batch_size, N_hn), dtype=torch.float)
        x = torch.FloatTensor([[-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50]])
        #already calculated delta_sigmoid values
        y = torch.FloatTensor([[4.53958e-05, 0.247517, 0.228784, 0.25, 0.249376, 0.235004, 0.104994, 1.92875e-22], 
                                [4.53958e-05, 0.247517, 0.228784, 0.25, 0.249376, 0.235004, 0.104994, 1.92875e-22]])
        grads = activation.delta_sigmoid(x)
        assert isinstance(grads, torch.FloatTensor)
        assert grads.size() == x.size() #torch.Size([batch_size, N_hn])
        precision = 0.0009
        self.assertTrue(torch.le(torch.abs(grads - y), precision).all())

    def test_softmax(self):
        batch_size = 2
        N_out = 8

        x = torch.FloatTensor([[-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-10, -0.2, -0.6, 0, 0.1, 0.5, 2, 50]])
        #already applied softmax values
        y = torch.FloatTensor([[8.7565e-27, 1.57913e-22, 1.05852e-22, 1.92875e-22, 2.1316e-22, 3.17997e-22, 1.42516e-21, 1],[8.7565e-27, 1.57913e-22, 1.05852e-22, 1.92875e-22, 2.1316e-22, 3.17997e-22, 1.42516e-21, 1]])
        outputs = activation.softmax(x)
        assert isinstance(outputs, torch.FloatTensor)
        assert outputs.size() == torch.Size([batch_size, N_out])
        precision = 0.0009
        self.assertTrue(torch.le(torch.abs(outputs - y), precision).all())

if __name__ == '__main__':
    unittest.main()