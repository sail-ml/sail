from test_case import *
import numpy as np
import sail
import time
import unittest, random

class ReshapeTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base_reshape(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3)]
        times = []
        for c in choices:
            for i in range(len(c)):
                c_ = list(c)
                random.shuffle(c_)
                arr1 = np.random.uniform(0, 1, (c_))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                
                t = time.time()
                x3 = sail.reshape(x1, c) 
                times.append(time.time() - t)
                arr3 = np.reshape(arr1, c) 

                self.assert_eq_np_sail(arr3, x3)
        return

