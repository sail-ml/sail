from test_case import *
import numpy as np
import sail
import time
import unittest

elementwise_options = [(12,), (32, 16), (3, 14, 2), (8, 12, 12, 2), (3, 1, 5, 6), (4, 14)]
broadcasted_options = [(32, 16), (3, 14, 2), (8, 12, 12, 2), (3, 1, 5, 6), (13, 4)]
unary_elementwise_options = [(12,), (32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
unary_broadcasted_options = [(32, 12), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]
grad_options = [(3), (32, 3, 5), (30), (12), (2, 33, 2, 5)]

class AddTest(UnitTest):

    @requires_grad_decorator
    def test_base_add(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            arr2 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            # t = time.time()
            x3 = sail.add(x1, x2) 
            # times.append(time.time() - t)
            arr3 = arr1 + arr2 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return
    @requires_grad_decorator
    def test_base_add_int(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1 + x1 + 2
            arr3 = 1 + arr1 + 2

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return
    @requires_grad_decorator
    def test_base_add_float(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1.2 + x1 + 1.3
            arr3 = 1.2 + arr1 + 1.3 

            self.assert_eq_np_sail(arr3, x3, eps=1e-8)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast_add_int(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)

                x3 = 1 + x1 + 2
                arr3 = 1 + arr1 + 2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return
    @requires_grad_decorator
    def test_broadcast_add_float(self, rq):
        choices = [(2, 8)]#broadcasted_options[:1]
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)
                
                x3 = 1.1 + x1 + 2.2
                arr3 = 1.1 + arr1 + 2.2

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
                
                arr1 = np.random.uniform(0, 1, (c)).astype(np.float32)
                c[i] = 1
                arr2 = np.random.uniform(0, 1, (c)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.add(x1, x2) 
                times.append(time.time() - t)
                arr3 = arr1 + arr2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    

    @dtype_decorator
    def test_dtype(self, dtype1, dtype2):
        choices = elementwise_options
        times = []
        for c in choices:
            x1 = sail.random.uniform(10, 20, c).astype(dtype1[0])
            x2 = sail.random.uniform(10, 20, c).astype(dtype2[0])
            arr1 = x1.numpy()
            arr2 = x2.numpy()

            x3 = x1 + x2 
            arr3 = arr1 + arr2 

            self.assert_eq_np_sail(arr3, x3)

            
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

    @requires_grad_decorator
    def test_base(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c)).astype(np.float32)
            arr2 = np.random.uniform(0, 1, (c)).astype(np.float32)
            
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

    @dtype_decorator
    def test_dtype(self, dtype1, dtype2):
        choices = elementwise_options
        times = []
        for c in choices:
            x1 = sail.random.uniform(10, 20, c).astype(dtype1[0])
            x2 = sail.random.uniform(10, 20, c).astype(dtype2[0])
            arr1 = x1.numpy()
            arr2 = x2.numpy()

            x3 = x1 - x2 
            arr3 = arr1 - arr2 

            self.assert_eq_np_sail(arr3, x3)

    @requires_grad_decorator
    def test_base_int(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1 - x1 - 2
            arr3 = 1 - arr1 - 2

            self.assert_eq_np_sail(arr3, x3, eps=1e-7)
            self.assert_eq(x3.requires_grad, rq)
        return
    @requires_grad_decorator
    def test_base_float(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1.2 - x1 - 1.3
            arr3 = 1.2 - arr1 - 1.3 

            self.assert_eq_np_sail(arr3, x3, eps=1e-7)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast_int(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b))#.astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)

                x3 = 1 - x1 - 2
                arr3 = 1 - arr1 - 2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return
    @requires_grad_decorator
    def test_broadcast_float(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b))#.astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)

                x3 = 1.1 - x1 - 2.2
                arr3 = 1.1 - arr1 - 2.2

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

#     # UnitTest._test_registry.append(AddTest)
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

    @dtype_decorator
    def test_dtype(self, dtype1, dtype2):
        choices = elementwise_options[:1]
        times = []
        for c in choices:
            x1 = sail.random.uniform(10, 20, c).astype(dtype1[0])
            x2 = sail.random.uniform(10, 20, c).astype(dtype2[0])
            arr1 = x1.numpy()
            arr2 = x2.numpy()

            x3 = x1 * x2 
            arr3 = arr1 * arr2 

            self.assert_eq_np_sail(arr3, x3)

    @requires_grad_decorator
    def test_base_int(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1 * x1 * 2
            arr3 = 1 * arr1 * 2 


            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return
    @requires_grad_decorator
    def test_base_float(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1.2 * x1 * 1.3
            arr3 = 1.2 * arr1 * 1.3 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast_int(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)

                x3 = 1 * x1 * 2
                arr3 = 1 * arr1 * 2 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return
    @requires_grad_decorator
    def test_broadcast_float(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)
                
                x3 = 1.1 * x1 * 2.2
                arr3 = 1.1 * arr1 * 2.2

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

            self.assert_eq_np_sail(arr3, x3, eps=1e-5)
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

                self.assert_eq_np_sail(arr3, x3, eps=1e-5)
                self.assert_eq(x3.requires_grad, rq)

        return

    @requires_grad_decorator
    def test_base_int(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1 / x1 / 2
            arr3 = 1 / arr1 / 2 


            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return
    @requires_grad_decorator
    def test_base_float(self, rq):
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))#.astype(np.float32)
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            x3 = 1.2 / x1 / 1.3
            arr3 = 1.2 / arr1 / 1.3 

            self.assert_eq_np_sail(arr3, x3)
            self.assert_eq(x3.requires_grad, rq)
        return

    @requires_grad_decorator
    def test_broadcast_int(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)

                x3 = 1 / x1 / 2
                arr3 = 1 / arr1 / 2 


                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return
    @requires_grad_decorator
    def test_broadcast_float(self, rq):
        choices = broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
        
                b[i] = 1
                arr1 = np.random.uniform(0, 1, (b)).astype(np.float32)
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x1 = sail.broadcast_to(x1, c)
                arr1 = np.broadcast_to(arr1, c)
                
                x3 = 1.2 / x1 / 1.3
                arr3 = 1.2 / arr1 / 1.3 


                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    @dtype_decorator
    def test_dtype(self, dtype1, dtype2):
        choices = [(2, 2)]#elementwise_options
        times = []
        for c in choices:
            x1 = sail.random.uniform(1, 2, c).astype(dtype1[0])
            x2 = sail.random.uniform(1, 2, c).astype(dtype2[0])
            arr1 = x1.numpy()
            arr2 = x2.numpy()

            x3 = x1 / x2 
            arr3 = arr1 / arr2 

            self.assert_eq_np_sail(arr3, x3)

    def test_broadcast2(self):
        a = sail.random.uniform(1, 2, (10, 1, 45))
        b = sail.random.uniform(1, 2, (10, 12, 45))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b, eps=1e-5)

        a = sail.random.uniform(1, 2, (10, 12))
        b = sail.random.uniform(1, 2, (10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b, eps=1e-5)

        a = sail.random.uniform(1, 2, (1, 10, 12))
        b = sail.random.uniform(1, 2, (3, 10, 1))

        a_np = a.numpy()
        b_np = b.numpy()

        self.assert_eq_np_sail(a_np / b_np, a / b, eps=1e-5)

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
        choices = elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(1, 2, (c))
            arr2 = np.random.uniform(1, 2, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            x2 = sail.Tensor(arr2, requires_grad=rq)
            
            t = time.time()
            x3 = sail.power(x1, x2) 
            times.append(time.time() - t)
            arr3 = np.power(arr1, arr2) 

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
                
                arr1 = np.random.uniform(1, 2, (c))
                c[i] = 1
                arr2 = np.random.uniform(1, 2, (c))
                
                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                
                t = time.time()
                x3 = sail.power(x1, x2) 
                times.append(time.time() - t)
                arr3 = np.power(arr1, arr2) 

                self.assert_eq_np_sail(arr3, x3)
                self.assert_eq(x3.requires_grad, rq)

        return

    def test_grad(self):

        def forward(a, b):
            c = sail.power(a, b)
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

class ExpTest(UnitTest):

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

    def test_integer(self):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c)).astype(np.int32)
            
            x1 = sail.Tensor(arr1)
            
            x3 = sail.exp(x1) 
            arr3 = np.exp(arr1) 

            self.assert_eq_np_sail(arr3, x3, eps=1e-6)
        return

    def test_broadcast(self):
        choices = unary_broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = 1
                arr1 = np.random.uniform(1, 2, (b))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.broadcast_to(x1, c)

                arr2 = np.broadcast_to(arr1, c)
                
                t = time.time()
                x3 = sail.exp(x2) 
                times.append(time.time() - t)
                arr3 = np.exp(arr2) 

                self.assert_eq_np_sail(arr3, x3, eps=1e-6)
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

class NegateTest(UnitTest):

    @requires_grad_decorator
    def test_base(self, rq):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, (c))
            
            x1 = sail.Tensor(arr1, requires_grad=rq)
            
            t = time.time()
            x3 = -x1 
            times.append(time.time() - t)
            arr3 = -arr1 

            self.assert_eq_np_sail(arr3, x3, eps=1e-6)
            self.assert_eq(x3.requires_grad, rq)
        return

    def test_broadcast(self):
        choices = unary_broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = 1
                arr1 = np.random.uniform(1, 2, (b))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.broadcast_to(x1, c)

                arr2 = np.broadcast_to(arr1, c)
                
                t = time.time()
                x3 = -x2
                times.append(time.time() - t)
                arr3 = -arr2

                self.assert_eq_np_sail(arr3, x3, eps=1e-6)
        return


    def test_grad(self):
        choices = unary_elementwise_options
        times = []

        def forward(a):
            c = -a
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

            self.assert_eq_np_sail(arr3, x3, eps=1e-6)
            self.assert_eq(x3.requires_grad, rq)
        return


    def test_broadcast(self):
        choices = unary_broadcasted_options
        times = []
        for c in choices:
            for i in range(len(c)):
                b = list(c)
                b[i] = 1
                arr1 = np.random.uniform(1, 2, (b))
                
                x1 = sail.Tensor(arr1, requires_grad=False)
                x2 = sail.broadcast_to(x1, c)

                arr2 = np.broadcast_to(arr1, c)
                
                t = time.time()
                x3 = sail.log(x2) 
                times.append(time.time() - t)
                arr3 = np.log(arr2) 

                self.assert_eq_np_sail(arr3, x3, eps=1e-6)
                # self.assert_eq(x3.requires_grad, rq)
        return

    def test_integer(self):
        choices = unary_elementwise_options
        times = []
        for c in choices:
            arr1 = np.random.uniform(1, 50000, (c)).astype(np.int16)
            
            x1 = sail.Tensor(arr1)
            
            x3 = sail.log(x1) 
            arr3 = np.log(arr1) 

            self.assert_eq_np_sail(arr3.astype(x3.numpy().dtype), x3, eps=1e-1)
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

#         return