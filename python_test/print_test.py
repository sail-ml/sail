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


        x = np.ones((32)).astype(np.float64)
        x *= 0.00000002
        x[16:] *= 1.3

        comparison = '''
tensor([0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002
        0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000002 0.00000003 0.00000003 0.00000003 0.00000003
        0.00000003 0.00000003 0.00000003 0.00000003 0.00000003 0.00000003 0.00000003 0.00000003 0.00000003 0.00000003
        0.00000003 0.00000003], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)

        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32)).astype(np.float64)
        x *= 100000000000
        x[16:] *= 1.3

        comparison = '''
tensor([1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11
        1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11
        1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11 1.30000000e+11
        1.30000000e+11 1.30000000e+11], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)

        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32, 12, 4)).astype(np.float64)
        x *= 100000000000
        x[16:] *= 0.00000002

        comparison = '''
tensor([[[1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]                                                                                                                                 
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         ...

         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]]

        [[1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         ...

         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]]

        [[1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         ...

         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]
         [1.00000000e+11 1.00000000e+11 1.00000000e+11 1.00000000e+11]]

        ...



        [[2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         ...

         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]]

        [[2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         ...

         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]]

        [[2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         ...

         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]
         [2.00000000e+03 2.00000000e+03 2.00000000e+03 2.00000000e+03]]], shape=(32, 12, 4))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)

        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

    def test_negative_print(self):
        x = np.ones((32, 32))
        x *= -0.001
        x[16:] *= 1.3

        comparison = '''
tensor([[-0.00100000 -0.00100000 -0.00100000 ...  -0.00100000
         -0.00100000 -0.00100000]
        [-0.00100000 -0.00100000 -0.00100000 ...  -0.00100000
         -0.00100000 -0.00100000]
        [-0.00100000 -0.00100000 -0.00100000 ...  -0.00100000
         -0.00100000 -0.00100000]
        ...

        [-0.0013     -0.0013     -0.0013     ...  -0.0013
         -0.0013     -0.0013    ]
        [-0.0013     -0.0013     -0.0013     ...  -0.0013
         -0.0013     -0.0013    ]
        [-0.0013     -0.0013     -0.0013     ...  -0.0013
         -0.0013     -0.0013    ]], shape=(32, 32))'''

        comparison = comparison[1:]
        comparison = comparison
        
        x = sail.Tensor(x)
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32)).astype(np.float64)
        x *= -0.00000002
        x[16:] *= 1.3

        comparison = '''
tensor([-0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002
        -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000002 -0.00000003 -0.00000003 -0.00000003 -0.00000003
        -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003 -0.00000003
        -0.00000003 -0.00000003], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32)).astype(np.float64)
        x *= -100000000000
        x[16:] *= 1.3

        comparison = '''
tensor([-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11
        -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11
        -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11 -1.30000000e+11
        -1.30000000e+11 -1.30000000e+11], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32, 12, 4)).astype(np.float64)
        x *= -100000000000
        x[16:] *= 0.00000002

        comparison = '''
tensor([[[-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]                                                                                                                             
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         ...

         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]]

        [[-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         ...

         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]]

        [[-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         ...

         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]
         [-1.00000000e+11 -1.00000000e+11 -1.00000000e+11 -1.00000000e+11]]

        ...



        [[-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         ...

         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]]

        [[-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         ...

         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]]

        [[-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         ...

         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]
         [-2.00000000e+03 -2.00000000e+03 -2.00000000e+03 -2.00000000e+03]]], shape=(32, 12, 4))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)

        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

    def test_general(self):
        x = np.ones((32, 32)).astype(np.int32)
        x *= 10
        x[16:] *= 20

        comparison = '''
tensor([[ 10  10  10 ...   10
          10  10]
        [ 10  10  10 ...   10
          10  10]
        [ 10  10  10 ...   10
          10  10]
        ...

        [200 200 200 ...  200
         200 200]
        [200 200 200 ...  200
         200 200]
        [200 200 200 ...  200
         200 200]], shape=(32, 32))'''

        comparison = comparison[1:]
        comparison = comparison
        
        x = sail.Tensor(x)
      
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32)).astype(np.float64)
        x *= -25.6
        x[16:] *= 13.8

        comparison = '''
tensor([ -25.6   -25.6   -25.6   -25.6   -25.6   -25.6   -25.6   -25.6   -25.6   -25.6
         -25.6   -25.6   -25.6   -25.6   -25.6   -25.6  -353.28 -353.28 -353.28 -353.28
        -353.28 -353.28 -353.28 -353.28 -353.28 -353.28 -353.28 -353.28 -353.28 -353.28
        -353.28 -353.28], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)
        
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones((32)).astype(np.int32)
        x *= -1000000000

        comparison = '''
tensor([-1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000
        -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000
        -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000 -1000000000
        -1000000000 -1000000000], shape=(32))'''

        comparison = comparison[1:]
        
        x = sail.Tensor(x)
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones(1)

        comparison = '''
tensor([[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]], shape=(10, 10))'''

        comparison = comparison[1:]
        x = sail.broadcast_to(sail.Tensor(x), (10, 10))
        
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.ones(10)

        comparison = '''
tensor([[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]], shape=(10, 10))'''

        comparison = comparison[1:]
        x = sail.broadcast_to(sail.Tensor(x), (10, 10))
        
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.array([1.25234312345])

        comparison = '''
tensor([[1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]
        [1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312 1.25234312]], shape=(10, 10))'''

        comparison = comparison[1:]
        x = sail.broadcast_to(sail.Tensor(x), (10, 10))
        
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        x = np.array([1.25234312345])

        comparison = '''tensor([1.25234312], shape=(1))'''

        x = sail.Tensor(x)
        
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))

        comparison = '''
tensor([ True False  True  True  True False False False  True  True
        False False  True  True False False  True  True False  True
         True False False False False False  True], shape=(27))'''

        comparison = comparison[1:]
        x = sail.Tensor(np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0,0,0,0,1])).astype(sail.bool_)
        self.assert_eq(x.__repr__().replace(" ", ""), comparison.replace(" ", ""))
