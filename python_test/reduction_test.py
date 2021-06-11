from test_case import *
import numpy as np
import sail
import time
import unittest

choices = [
    {"shape": (12, 3, 14), "axis": None, "keepdims": False, "result_shape": (1,)},
    {"shape": (12, 14), "axis": None, "keepdims": False, "result_shape": (1,)},
    {"shape": (14,), "axis": None, "keepdims": False, "result_shape": (1,)},
    {"shape": (12, 3, 14), "axis": -1, "keepdims": False, "result_shape": (12, 3)},
    {"shape": (2, 18, 3), "axis": 2, "keepdims": False, "result_shape": (2, 18)},
    {"shape": (4, 5, 6), "axis": 0, "keepdims": False, "result_shape": (5, 6)},
    {"shape": (17, 1), "axis": 1, "keepdims": False, "result_shape": (17,)},
    {"shape": (14, 18, 14, 3), "axis": -2, "keepdims": False, "result_shape": (14, 18, 3)},
    {"shape": (3, 4, 5, 1, 3), "axis": None, "keepdims": False, "result_shape": (1,)},
    {"shape": (3, 4, 5, 1, 3), "axis": 2, "keepdims": False, "result_shape": (3, 4, 1, 3)},
    {"shape": (12, 3, 14), "axis": 1, "keepdims": True, "result_shape": (12, 1, 14)},
    {"shape": (12, 3, 14), "axis": 2, "keepdims": True, "result_shape": (12, 3, 1)},
    {"shape": (12, 3, 14), "axis": 0, "keepdims": True, "result_shape": (1, 3, 14)},
    {"shape": (12, 3), "axis": -1, "keepdims": True, "result_shape": (12, 1)},
]

grad_choices = [(32, 2), (12, 14), (100)]

class SumTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_sum(self, rq):
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c["shape"]))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = sail.sum(x1, c['axis'], keepdims=c['keepdims']) 
            arr3 = np.sum(arr1, c["axis"], keepdims=c['keepdims'])

            self.assert_eq(x3.shape, c["result_shape"])
            self.assert_eq_np_sail(arr3, x3, 1e-7)
            self.assert_eq(x3.requires_grad, rq)
        return

    def test_sum_grad(self):
        def forward(a):
            y = sail.sum(a)
            return y 
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c["shape"]))
            
            dic = {
                "a": arr1,
            }

            diff = check_gradients_vector(forward, dic, eps=1e-6)
            assert diff < 1e-6
        return

class MeanTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_mean(self, rq):
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c["shape"]))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = sail.mean(x1, c["axis"], keepdims=c['keepdims'])
            arr3 = np.mean(arr1, c["axis"], keepdims=c['keepdims'])

            self.assert_eq(x3.shape, c["result_shape"])
            self.assert_eq_np_sail(arr3, x3, 1e-7)
            self.assert_eq(x3.requires_grad, rq)
        return

class MaxTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_max(self, rq):
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c["shape"]))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = sail.max(x1, c["axis"], keepdims=c["keepdims"])
            arr3 = np.max(arr1, c["axis"], keepdims=c["keepdims"])

            self.assert_eq(x3.shape, c["result_shape"])
            self.assert_eq_np_sail(arr3, x3, 1e-7)
            self.assert_eq(x3.requires_grad, rq)
        return

    def test_max_grad(self):
        def forward(a):
            y = sail.max(a)
            return y 
        times = []
        for c in grad_choices:
            arr1 = np.random.uniform(1, 10, c)
            
            dic = {
                "a": arr1,
            }

            diff = check_gradients_vector(forward, dic, eps=1e-6)
            assert diff < 1e-6
        return
