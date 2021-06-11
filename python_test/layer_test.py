from test_case import *
import numpy as np
import sail
import time
import unittest, random

class LinearLayerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_no_bias(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca)).astype(np.float32)

                x1 = sail.Tensor(arr1, requires_grad=False)

                lin = sail.modules.Linear(cb[0], cb[1], use_bias=False)

                y = lin(x1)
                y2 = np.matmul(arr1, lin.weights.numpy())

                self.assert_eq_np_sail(y2, y, eps=5e-7)
                
        return

    def test_bias(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca)).astype(np.float32)

                x1 = sail.Tensor(arr1, requires_grad=False)

                lin = sail.modules.Linear(cb[0], cb[1], use_bias=True)

                y = lin(x1)
                y2 = np.matmul(arr1, lin.weights.numpy()) + lin.bias.numpy()

                self.assert_eq_np_sail(y2, y, eps=5e-7)
        return
