from test_case import *
import numpy as np
import sail
import time
import unittest, random
import tensorflow as tf 


class Math(UnitTest):

    def test_1(self):

        def forward(a, b, d, f):
            c = (a / b) + (d * f)

            h = sail.exp(c)
            g = h - sail.max(h, 1, True)
            i = sail.sum(h)

            return i 

        dic = {
            "a": np.random.uniform(1, 2, (12, 3)),
            "b": np.random.uniform(1, 1.1876, (12, 3)),
            "d": np.random.uniform(1, 1.5, (12, 3)),
            "f": np.random.uniform(0.5, 1.225, (12, 3))
        }

        self.assert_true(check_gradients_vector(forward, dic))

    def test_1(self):

        def forward(a, b, d, f):
            c = (a / b) + (d * f)

            h = sail.exp(c)
            g = h - sail.max(h, 1, True) + d - c
            i = sail.sum(h)

            return i 

        dic = {
            "a": np.random.uniform(1, 2, (12, 3)),
            "b": np.random.uniform(1, 1.1876, (12, 3)),
            "d": np.random.uniform(1, 1.5, (12, 3)),
            "f": np.random.uniform(0.5, 1.225, (12, 3))
        }

        self.assert_true(check_gradients_vector(forward, dic))




    