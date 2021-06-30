from test_case import *
import numpy as np
import sail
import time
import unittest

elementwise_options = [(12,), (512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
broadcasted_options = [(512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_elementwise_options = [(12,), (32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_broadcasted_options = [(32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
grad_options = [(32, 3, 5), (3), (1), (2, 33, 2, 5)]

class CastTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @dtype_decorator
    def test_base(self, dtype1, dtype2):
        choices = elementwise_options
        times = []
        for c in choices:
            x1 = sail.random.uniform(1, 100000, c)
            x2 = x1.astype(dtype1[0])
            arr1 = x2.numpy()

            self.assert_eq(arr1.dtype, dtype1[1])
            self.assert_eq_np_sail(arr1, x2)

            x2 = x1.astype(dtype2[0])
            arr1 = x2.numpy()
            self.assert_eq(arr1.dtype, dtype2[1])
            self.assert_eq_np_sail(arr1, x2)
        #     arr1 = np.random.uniform(0, 1, (c))
        #     arr2 = np.random.uniform(0, 1, (c))
            
        #     x1 = sail.Tensor(arr1, requires_grad=rq)
        #     x2 = sail.Tensor(arr2, requires_grad=rq)
            
        #     t = time.time()
        #     x3 = sail.add(x1, x2) 
        #     times.append(time.time() - t)
        #     arr3 = arr1 + arr2 

        #     self.assert_eq_np_sail(arr3, x3)
        #     self.assert_eq(x3.requires_grad, rq)
        # return
