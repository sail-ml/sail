from test_case import *
import numpy as np
import sail
import time
import unittest, random

class LinearLayerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_no_bias(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca)).astype(np.float32)

                x1 = sail.Tensor(arr1, requires_grad=False)

                lin = sail.modules.Linear(cb[0], cb[1], use_bias=False)

                y = lin(x1)
                y2 = np.matmul(arr1, lin.weights.numpy())

                self.assert_eq_np_sail(y2, y, eps=5e-6)
                self.assert_eq(y.requires_grad, True)
                
        return

    def test_bias(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca)).astype(np.float32)

                x1 = sail.Tensor(arr1, requires_grad=False)

                lin = sail.modules.Linear(cb[0], cb[1], use_bias=True)

                y = lin(x1)
                y2 = np.matmul(arr1, lin.weights.numpy()) + lin.biases.numpy()

                self.assert_eq_np_sail(y2, y, eps=5e-6)
                self.assert_eq(y.requires_grad, True)
        return

    def test_get_set(self):
        l = sail.modules.Linear(32, 64)
        weights = l.weights 
        biases = l.biases

        new_weights = sail.random.uniform(0, 0.01, (32, 64)) 
        new_biases = sail.random.uniform(0, 0.01, (32)) 

        input_ = sail.random.uniform(0, 1, (128, 32))

        y1 = l(input_)
        l.weights = new_weights
        l.biases = new_biases
        y2 = l(input_)

        self.assert_neq_np(y1.numpy(), y2.numpy())
        self.assert_true(l.weights.requires_grad)
        self.assert_true(l.biases.requires_grad)


class SigmoidLayerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, c).astype(np.float32)

            x1 = sail.Tensor(arr1, requires_grad=rq)

            lin = sail.modules.Sigmoid()

            y = lin(x1)
            y2 = 1/(1 + np.exp(-arr1))

            self.assert_eq_np_sail(y2, y, eps=5e-6)
            self.assert_eq(y.requires_grad, rq)
                
        return

class SoftmaxLayerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(0, 1, c).astype(np.float32)

            x1 = sail.Tensor(arr1, requires_grad=rq)

            lin = sail.modules.Softmax()

            y = lin(x1)
            y2 = np.exp(arr1) / np.sum(np.exp(arr1), 1, keepdims=True)

            self.assert_eq_np_sail(y2, y, eps=5e-6)
            self.assert_eq(y.requires_grad, rq)
                
        return


class ReLULayerTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    @requires_grad_decorator
    def test(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        times = []
        for c in choices:
            arr1 = np.random.uniform(-1, 1, c).astype(np.float32)

            x1 = sail.Tensor(arr1, requires_grad=rq)

            lin = sail.modules.ReLU()

            y = lin(x1)
            y2 = np.clip(arr1, 0, 100)

            self.assert_eq_np_sail(y2, y, eps=1e-8)
            self.assert_eq(y.requires_grad, rq)
                
        return

    def test_add_grad(self):
        times = []

        def forward(a):
            c = sail.modules.ReLU()(a)
            d = sail.sum(c)
            return d

        for c in [(3, 3), (12, 18), (2, 33), (32, 64)]:
            
            arr1 = np.random.uniform(-1, 1, (c))
            
            dic = {
                "a": arr1,
            }

            self.assert_true(check_gradients_vector(forward, dic, rtol=1e-2, atol=1e-4, eps=1e-8))

        return


class Conv2DLayerTest(UnitTest):

    def test_no_bias(self):
        img = sail.random.uniform(0, 1, (64, 4, 277, 277))
        lay = sail.modules.Conv2D(4, 32, 3, 1, use_bias=False)
        y = lay(img)
        self.assert_eq(y.shape, (64, 32, 277, 277))

        z = sail.sum(y)
        z.backward()


    def test_bias(self):
        img = sail.random.uniform(0, 1, (64, 4, 277, 277))
        lay = sail.modules.Conv2D(4, 32, 3, 1, use_bias=True)
        y = lay(img)
        self.assert_eq(y.shape, (64, 32, 277, 277))

        z = sail.sum(y)
        z.backward()