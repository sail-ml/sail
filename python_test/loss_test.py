from test_case import *
import numpy as np
import sail
import torch 
import time
import unittest, random

class SoftmaxCrossEntropyTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, c).astype(np.float32)
            targs = np.random.randint(0, c[1], c[0]).astype(np.int64)

            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(targs, requires_grad=False)

            lin = sail.losses.SoftmaxCrossEntropy()

            loss_sail = lin(x1, x2)

            t_loss_fcn = torch.nn.CrossEntropyLoss()
            t1 = torch.from_numpy(arr1)
            t2 = torch.from_numpy(targs)
            loss_torch = t_loss_fcn(t1, t2)

            self.assert_eq_np(loss_sail.numpy(), loss_torch.detach().numpy(), eps=1e-5)
                
        return

class MeanSquaredErrorTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, c).astype(np.float32)
            targs = np.random.uniform(0, 1, c).astype(np.float32)

            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(targs, requires_grad=rq)

            lin = sail.losses.MeanSquaredError()

            loss_sail = lin(x1, x2)

            loss_numpy = np.mean((arr1 - targs) ** 2)

            self.assert_eq_np(loss_sail.numpy(), loss_numpy, eps=1e-5)
                
        return

    def test_grad(self):

        def forward(a, b):
            c = sail.losses.MeanSquaredError()(a, b)
            return c

        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        for c in choices:
            
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            dic = {
                "a": arr1,
                "b": arr2
            }

            self.assert_true(check_gradients_vector(forward, dic))

        return
