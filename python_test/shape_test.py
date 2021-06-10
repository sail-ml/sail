from test_case import *
import numpy as np
import sail
import time
import unittest, random

class ReshapeTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base_reshape(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
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
                self.assert_eq(x3.shape, c)
        return

class ExpandDimsTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
        times = []
        for c in choices:
            for i in range(-len(c)-1, len(c)):
                c = list(c)
                arr1 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                
                t = time.time()
                x3 = sail.expand_dims(x1, i) 
                times.append(time.time() - t)
                arr3 = np.expand_dims(arr1, i) 


                self.assert_eq_np_sail(arr3, x3)
                c_ = list(c)
                a = i
                if (i < 0):
                    i += len(c) + 1
                if (i == len(c)):
                    c_.append(1)
                else:
                    c_.insert(i, 1)
                self.assert_eq(list(x3.shape), c_)
                
        return
    
class SqueezeTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
        times = []
        for c in choices:
            for i in range(-len(c)-1, len(c)):
                c = list(c)
                arr1 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                
                t = time.time()
                x3 = sail.expand_dims(x1, i) 
                times.append(time.time() - t)
                arr3 = np.expand_dims(arr1, i) 
               
                self.assert_eq_np_sail(arr3, x3)
                c_ = list(c)
                a = i
                if (i < 0):
                    i += len(c) + 1
                if (i == len(c)):
                    c_.append(1)
                else:
                    c_.insert(i, 1)
                self.assert_eq(list(x3.shape), c_)
                
        return
    
class TransposeTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3)]
        times = []
        for c in choices:
            for i in range(len(c)):
                cr = list(range(len(c)))
                random.shuffle(cr)

                fixed_shape = [c[r] for r in cr]

                arr1 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                
                x3 = sail.transpose(x1, cr) 
                arr3 = np.transpose(arr1, cr) 
               
                self.assert_eq_np_sail(arr3, x3)
               
                self.assert_eq(list(x3.shape), fixed_shape)
                
        return
    
