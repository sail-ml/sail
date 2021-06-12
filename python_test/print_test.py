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

class PrintTest(UnitTest):

    ## 1
    def test_print(self):
        x = np.ones((32, 32))
        x *= 0.001
        x[16:] *= 1.3

        comparison = '''
tensor([[0.00100000 0.00100000 0.00100000 ...  0.00100000
         0.00100000 0.00100000]
        [0.00100000 0.00100000 0.00100000 ...  0.00100000
         0.00100000 0.00100000]
        [0.00100000 0.00100000 0.00100000 ...  0.00100000
         0.00100000 0.00100000]
        ...

        [0.0013     0.0013     0.0013     ...  0.0013
         0.0013     0.0013    ]
        [0.0013     0.0013     0.0013     ...  0.0013
         0.0013     0.0013    ]
        [0.0013     0.0013     0.0013     ...  0.0013
         0.0013     0.0013    ]], shape=(32, 32))'''

        comparison = comparison[1:]
        comparison = comparison
        
        x = sail.Tensor(x)

        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))