from test_case import *
import numpy as np
import sail
import time
import unittest, random

elementwise_options = [(12,), (512, 128), (3, 14, 2), (8, 12, 12, 12), (1, 2, 3), (10,), (13, 3, 5, 2, 1, 7, 5), (3, 1, 5, 6), (13, 14)]
broadcasted_options = [(512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_elementwise_options = [(12,), (32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_broadcasted_options = [(32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
grad_options = [(32, 3, 5), (3), (1), (2, 33, 2, 5)]

class CatTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        choices = elementwise_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = random.randint(min(b), max(c) + 10)
                x1 = sail.random.uniform(10, 20, c)
                x2 = sail.random.uniform(10, 20, b)
                
                x3 = sail.cat([x1, x2], axis=i)

                arr3 = np.concatenate([x1.numpy(), x2.numpy()], axis=i)

                self.assert_eq_np_sail(arr3, x3)

    def test_broadcast(self):
        choices = elementwise_options
        times = []
        for c in choices:
            for i in range(0, len(c)):
                b = list(c)
                b[i] = random.randint(min(b), max(c) + 10)
                c_ = list(c)
                c_[i-1] = 1
                x1 = sail.random.uniform(10, 20, c_)
                x1 = sail.broadcast_to(x1, c)
                x2 = sail.random.uniform(10, 20, b)
                
                x3 = sail.cat([x1, x2], axis=i)

                arr3 = np.concatenate([x1.numpy(), x2.numpy()], axis=i)

                self.assert_eq_np_sail(arr3, x3)


class StackTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        choices = elementwise_options
        times = []
        for c in choices:
            for i in range(len(c)):
                x1 = sail.random.uniform(10, 20, c)
                x2 = sail.random.uniform(10, 20, c)
                
                x3 = sail.stack([x1, x2], axis=i)

                arr3 = np.stack([x1.numpy(), x2.numpy()], axis=i)

                self.assert_eq_np_sail(arr3, x3)

    def test_broadcast(self):
        choices = elementwise_options
        times = []
        for c in choices:
            for i in range(0, len(c)):
                b = list(c)
                c_ = list(c)
                c_[i-1] = 1
                x1 = sail.random.uniform(10, 20, c_)
                x1 = sail.broadcast_to(x1, c)
                x2 = sail.random.uniform(10, 20, c)
                
                x3 = sail.stack([x1, x2], axis=i)

                arr3 = np.stack([x1.numpy(), x2.numpy()], axis=i)

                self.assert_eq_np_sail(arr3, x3)


