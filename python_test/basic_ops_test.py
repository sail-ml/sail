from test_case import *
import numpy as np
import sail
import time
import unittest

elementwise_options = [(12,), (512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
broadcasted_options = [(512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_elementwise_options = [(12,), (32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_broadcasted_options = [(32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
grad_options = [(32, 3, 5), (30), (12), (2, 33, 2, 5)]

class AddTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base_add(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.add(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 + arr2 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast_add(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.add(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 + arr2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    def test_broadcast_add2(self):
        a = sail.random.uniform(0, 1, (10, 1, 45))
        b = sail.random.uniform(0, 1, (10, 12, 45))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np + b_np, a + b)

        a = sail.random.uniform(0, 1, (10, 12))
        b = sail.random.uniform(0, 1, (10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np + b_np, a + b)

        a = sail.random.uniform(0, 1, (1, 10, 12))
        b = sail.random.uniform(0, 1, (3, 10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np + b_np, a + b)

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

            self.assert_true(check_gradients_vector(forward, dic))

        return

class SubtractTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.subtract(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 - arr2 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.subtract(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 - arr2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)
        return

    def test_broadcast2(self):
        a = sail.random.uniform(0, 1, (10, 1, 45))
        b = sail.random.uniform(0, 1, (10, 12, 45))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np - b_np, a - b)

        a = sail.random.uniform(0, 1, (10, 12))
        b = sail.random.uniform(0, 1, (10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np - b_np, a - b)

        a = sail.random.uniform(0, 1, (1, 10, 12))
        b = sail.random.uniform(0, 1, (3, 10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np - b_np, a - b)

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

                self.assert_true(check_gradients_vector(forward, dic))


            return

class MultiplyTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.multiply(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 * arr2 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.multiply(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 * arr2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    def test_broadcast2(self):
        a = sail.random.uniform(0, 1, (10, 1, 45))
        b = sail.random.uniform(0, 1, (10, 12, 45))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np * b_np, a * b)

        a = sail.random.uniform(0, 1, (10, 12))
        b = sail.random.uniform(0, 1, (10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np * b_np, a * b)

        a = sail.random.uniform(0, 1, (1, 10, 12))
        b = sail.random.uniform(0, 1, (3, 10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np * b_np, a * b)

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

            self.assert_true(check_gradients_vector(forward, dic))

        return

class DivideTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.divide(x1, x2) 
            times.append(time.time() - t)
            arr3 = arr1 / arr2 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.divide(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 / arr2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    def test_broadcast2(self):
        a = sail.random.uniform(1, 2, (10, 1, 45))
        b = sail.random.uniform(1, 2, (10, 12, 45))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b)

        a = sail.random.uniform(1, 2, (10, 12))
        b = sail.random.uniform(1, 2, (10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b)

        a = sail.random.uniform(1, 2, (1, 10, 12))
        b = sail.random.uniform(1, 2, (3, 10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b)

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

            self.assert_true(check_gradients_vector(forward, dic))

        return

class PowerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            arr2 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.power(x1, x2) 
            times.append(time.time() - t)
            arr3 = np.power(arr1, arr2) 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    def broadcast_test(self):
        choices = broadcasted_options
        times = []
        for c in choices:
            c = list(c)
            for i in range(len(c)):
                
                arr1 = np.random.uniform(0, 1, (c))
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.power(x1, x2) 
                times.append(time.time() - t)
                arr3 = np.power(arr1, arr2) 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

class ExpTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            t = time.time()
            x3 = sail.exp(x1) 
            times.append(time.time() - t)
            arr3 = np.exp(arr1) 

            self.assert_eq_np_sail(arr3, x3, eps=1e-6)
            self.assert_eq(x3.requires_grad, rq)
        return

    def tets_boradcast(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.broadcast_to(x1, c)

                arr2 = np.broadcast_to(arr1, c)
                
                t = time.time()
                x3 = sail.exp(x2) 
                times.append(time.time() - t)
                arr3 = np.exp(arr2) 

                self.assert_eq_np_sail(arr3, x3, eps=1e-6)
                self.assert_eq(x3.requires_grad, rq)
        return


    def test_grad(self):
        choices = unary_elementwise_options
        times = []

        def forward(a):
            c = sail.exp(a)
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(1, 2, (c))
            
            dic = {
                "a": arr1,
            }
            self.assert_true(check_gradients_vector(forward, dic))

        return

class LogTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(1, 2, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            t = time.time()
            x3 = sail.log(x1) 
            times.append(time.time() - t)
            arr3 = np.log(arr1) 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return


    def tets_boradcast(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.broadcast_to(x1, c)

                arr2 = np.broadcast_to(arr1, c)
                
                t = time.time()
                x3 = sail.log(x2) 
                times.append(time.time() - t)
                arr3 = np.log(arr2) 

                self.assert_eq_np_sail(arr3, x3, eps=1e-6)
                self.assert_eq(x3.requires_grad, rq)
        return


    def test_grad(self):
        choices = unary_elementwise_options
        times = []

        def forward(a):
            c = sail.log(a)
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(1, 2, (c))
            
            dic = {
                "a": arr1,
            }
            self.assert_true(check_gradients_vector(forward, dic))

        return
        
        
class ClipTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test_base(self, rq):
        choices = unary_elementwise_options
        for c in choices:
            arr1 = np.random.normal(5, 2, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)

            arr2 = np.clip(arr1, 4, 6)
            x2 = sail.clip(x1, 4, 6)

            self.assert_eq_np_sail(arr2, x2)
            self.assert_eq(x2.requires_grad, rq)
        return

    def test_grad(self):
        times = []

        def forward(a):
            c = sail.clip(a, 4, 6)
            d = sail.sum(c)
            return d

        for c in grad_options:
            
            arr1 = np.random.uniform(3, 7, (c))
            
            dic = {
                "a": arr1,
            }
            self.assert_true(check_gradients_vector(forward, dic, rtol=1e-2, atol=1e-4, eps=1e-8))

        return