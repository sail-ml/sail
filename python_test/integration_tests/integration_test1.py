from test_case import *
import numpy as np
import sail
import time
import unittest, random
import tensorflow as tf 


class IntegrationTests1(UnitTest):

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

    def test_2(self):

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

    def test_3(self):

        a = sail.random.uniform(0, 1, (10, 20))
        anp = a.numpy()

        a = sail.reshape(a, (5, 40))
        a = sail.expand_dims(a, 1)

        b = sail.random.uniform(0, 1, (5, 1))
        bnp = b.numpy()

        b = a * a + b 

        anp = np.reshape(anp, (5, 40))
        anp = np.expand_dims(anp, 1)

        bnp = anp * anp + bnp 

        self.assert_eq_np_sail(bnp, b)

    def layer_inits(self):

        m = sail.modules.MaxPool2D(2)
        m = sail.modules.MaxPool2D((2, 2))
        m = sail.modules.MaxPool2D((2, 2))
        m = sail.modules.MaxPool2D((2, 2), (2, 2), padding_mode="valid")

        m = sail.modules.Conv2D(2, 3, 1)
        m = sail.modules.Conv2D(2, 3, (3, 3), 1)
        m = sail.modules.Conv2D(2, 3, (3, 3), (3, 3))

    def test_4(self):

        def forward(a, b, d, f):
            c = (a / b) + (d * f)
            c = c + a

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