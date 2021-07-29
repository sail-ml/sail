from test_case import *
import numpy as np
import sail
import time
import unittest, random

class PadTest(UnitTest):

    data = [
        {"shape": (3, 2, 3), "pad": ((1, 1))},
        {"shape": (3, 2, 3), "pad": ((2, 1), (3, 1), (1, 3))},
        {"shape": (5, 6, 1), "pad": ((0, 3), (10, 0), (0, 0))},
        {"shape": (3, 10), "pad": ((0, 0), (0, 0))},
        {"shape": (4, 10, 5, 10), "pad": ((0, 3), (1, 1), (0, 1), (1, 9))},
        {"shape": (4), "pad": ((0, 0))},
        {"shape": (4), "pad": ((128, 1))},
    ]

    def test_base(self):

        for i in PadTest.data:
            shape = i["shape"]
            pad = i["pad"]

            a = sail.random.uniform(0, 1, shape)
            a_padded = sail.pad(a, pad)
            a_n_padded = np.pad(a.numpy(), pad)

            self.assert_eq_np_sail(a_n_padded, a_padded)


