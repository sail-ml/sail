from test_case import *
import numpy as np
import sail
import time
import unittest, random

choices = [(12,), (512, 128), (3, 14, 2), (8, 12, 12, 12), (1, 2, 3), (10,), (13, 3, 5, 2, 1, 7, 5), (3, 1, 5, 6), (13, 14)]

class EqTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        times = []
        for c in choices:
            x1 = sail.random.uniform(0, 1, c)
            x2 = sail.random.uniform(0, 1, c)

            x3 = x1 == x2 

            arr3 = x1.numpy() == x2.numpy()

            self.assert_eq_np_sail(arr3, x3)

class GTTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        times = []
        for c in choices:
            x1 = sail.random.uniform(0, 1, c)
            x2 = sail.random.uniform(0, 1, c)

            x3 = x1 > x2 

            arr3 = x1.numpy() > x2.numpy()

            self.assert_eq_np_sail(arr3, x3)

class LTTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        times = []
        for c in choices:
            x1 = sail.random.uniform(0, 1, c)
            x2 = sail.random.uniform(0, 1, c)

            x3 = x1 < x2 

            arr3 = x1.numpy() < x2.numpy()

            self.assert_eq_np_sail(arr3, x3)

class GTETest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        times = []
        for c in choices:
            x1 = sail.random.uniform(0, 1, c)
            x2 = sail.random.uniform(0, 1, c)

            x3 = x1 >= x2 

            arr3 = x1.numpy() >= x2.numpy()

            self.assert_eq_np_sail(arr3, x3)

class LTETest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        times = []
        for c in choices:
            x1 = sail.random.uniform(0, 1, c)
            x2 = sail.random.uniform(0, 1, c)

            x3 = x1 <= x2 

            arr3 = x1.numpy() <= x2.numpy()

            self.assert_eq_np_sail(arr3, x3)

