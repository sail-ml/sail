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

class AddTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base_add(self):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.add(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 + arr2 

            self.assert_eq_np_sail(arr3, x3)
        return

    def test_broadcast_add(self):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.Tensor(arr2, requires_grad=False)
                
                t = time.time()
                x3 = sail.add(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 + arr2 

                self.assert_eq_np_sail(arr3, x3)

        return

    def test_add_grad(self):
        choices = broadcasted_options
        times = []

        def forward(a, b):
            c = sail.add(a, b)
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            dic = {
                "a": arr1,
                "b": arr2
            }

            diff = check_gradients_vector(forward, dic)
            assert diff < 1e-6

        return

class SubtractTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.subtract(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 - arr2 

            self.assert_eq_np_sail(arr3, x3)
        return

    def test_broadcast(self):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.Tensor(arr2, requires_grad=False)
                
                t = time.time()
                x3 = sail.subtract(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 - arr2 

                self.assert_eq_np_sail(arr3, x3)

        return

        def test_grad(self):
            choices = broadcasted_options
            times = []

            def forward(a, b):
                c = a - b 
                d = sail.sum(c)
                return d

            for c in grad_options:
                
                arr1 = np.random.uniform(0, 1, (c))
                arr2 = np.random.uniform(0, 1, (c))
                
                dic = {
                    "a": arr1,
                    "b": arr2
                }

                diff = check_gradients_vector(forward, dic)
                assert diff < 1e-6

            return

class MultiplyTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.multiply(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 * arr2 

            self.assert_eq_np_sail(arr3, x3)
        return

    def test_broadcast(self):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.Tensor(arr2, requires_grad=False)
                
                t = time.time()
                x3 = sail.multiply(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 * arr2 

                self.assert_eq_np_sail(arr3, x3)

        return

    def test_grad(self):
        choices = broadcasted_options
        times = []

        def forward(a, b):
            c = a * b 
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            dic = {
                "a": arr1,
                "b": arr2
            }

            diff = check_gradients_vector(forward, dic)
            assert diff < 1e-6

        return

class DivideTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.divide(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 / arr2 

            self.assert_eq_np_sail(arr3, x3)
        return

    def test_broadcast(self):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.Tensor(arr2, requires_grad=False)
                
                t = time.time()
                x3 = sail.divide(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 / arr2 

                self.assert_eq_np_sail(arr3, x3)

        return

    def test_grad(self):
        choices = broadcasted_options
        times = []

        def forward(a, b):
            c = a / b 
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(1, 2, (c))
            arr2 = np.random.uniform(1, 2, (c))
            
            dic = {
                "a": arr1,
                "b": arr2
            }

            diff = check_gradients_vector(forward, dic)
            assert diff < 1e-5, (diff, 1e-5)

        return

class PowerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)

    def test_base(self):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.power(x1, x2) 
            times.append(time.time() - t)
            arr3 = np.power(arr1, arr2) 

            self.assert_eq_np_sail(arr3, x3)
        return
