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

    def test_catch_error(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
        times = []
        for c in choices:
            c_ = list(c)
            random.shuffle(c_)
            c_[-1] = c_[-1] * 2
            arr1 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            
            try:
                x3 = sail.reshape(x1, c_) 
            except sail.DimensionError as e:
                self.assert_eq(True, True)
                return 
            
            self.assert_eq(False, True)
               
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
                
                arr3 = np.expand_dims(arr1, i) 
                t = time.time()
                x3 = sail.expand_dims(x1, i) 
                times.append(time.time() - t)


                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(arr3.shape, x3.shape)
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

    def test_catch_error(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            
            try:
                x3 = sail.expand_dims(x1, 10) 
            except sail.DimensionError as e:
                try:
                    x3 = sail.expand_dims(x1, -10) 
                except sail.DimensionError as e:
                    self.assert_eq(True, True)
                    return 
            
            self.assert_eq(False, True)
               
        return
    
class SqueezeTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = [(12, 13, 3), (2, 5), (18, 15), (1, 3, 2, 3), (2,)]
        arrs = []
        for c in choices:
            temp = []
            for i in range(-len(c)-1, len(c)):
                x = np.expand_dims(np.random.uniform(0, 1, (c)), i)
                temp.append(x)
            arrs.append(temp)
        times = []
        for a, c in zip(arrs, choices):
            z = 0
            for i in range(-a[0].ndim, a[0].ndim-1):
                arr1 = a[z]
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                
                arr3 = np.squeeze(arr1, i) 
                x3 = sail.squeeze(x1, i) 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(arr3.shape, x3.shape)
                z += 1
        return

    def test_base(self):
        x = sail.random.uniform(0, 1, (3, 2, 3, 1))
        try:
            sail.squeeze(x, -2)
        except sail.SailError:
            self.assert_eq(True, True)

        x = sail.random.uniform(0, 1, (3, 2, 30, 1))
        try:
            x = sail.squeeze(x, -1)
            sail.squeeze(x, -1)
        except sail.SailError:
            self.assert_eq(True, True)
        
    
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
    
