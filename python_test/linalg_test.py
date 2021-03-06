from test_case import *
import numpy as np
import sail
import time
import unittest, random

class MatmulTest(UnitTest):

    @requires_grad_decorator
    def test_base(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca))
                arr2 = np.random.uniform(0, 1, (cb))

                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)

                x3 = sail.matmul(x1, x2)
                arr3 = np.matmul(arr1, arr2)

                self.assert_eq(x3.shape, verif_shape)
                self.assert_eq_np_sail(arr3, x3, eps=1e-7)
                self.assert_eq(x3.requires_grad, rq)

        return

    def test_integer(self):
        a = sail.random.uniform(0, 11, (10, 20)).astype(sail.int32)
        b = sail.random.uniform(0, 11, (20, 10)).astype(sail.int32)

        c = sail.matmul(a, b)
        a = a.numpy()
        b = b.numpy()

        c2 = np.matmul(a, b)

        self.assert_eq_np_sail(c2, c, eps=1e-7)



    def test_error_base(self):
        a = sail.random.uniform(0, 1, (1))
        b = sail.random.uniform(1, 2, (1))
        self.assert_throws(sail.matmul, (a, b), sail.SailError)

        a = sail.random.uniform(0, 1, (3, 4, 5))
        b = sail.random.uniform(1, 2, (5, 2))
        self.assert_throws(sail.matmul, (a, b), sail.SailError)


        a = sail.random.uniform(0, 1, (3, 5))
        b = sail.random.uniform(1, 2, (3, 5))
        self.assert_throws(sail.matmul, (a, b), sail.SailError)
        
    
    def test_grad(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []

        def forward(a, b):
            c = sail.matmul(a, b)
            d = sail.sum(c)
            return d

        for ca, cbs in zip(choices, choices_2):
            for cb in cbs[:1]:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca))
                arr2 = np.random.uniform(0, 1, (cb))

                dic = {
                    "a": arr1,
                    "b": arr2
                }

                self.assert_true(check_gradients_vector(forward, dic))

        return
    
    def test_grad_with_vec1(self):
        choices = [(3), (18), (33), (64)]
        choices2 = [(3, 3), (18, 32), (33, 12), (64, 2)]
        times = []

        def forward(a, b):
            c = sail.matmul(a, b)
            d = sail.sum(c)
            return d

        for ca, cb in zip(choices, choices2):
            arr1 = np.random.uniform(0, 1, (ca))
            arr2 = np.random.uniform(0, 1, (cb))

            dic = {
                "a": arr1,
                "b": arr2
            }

            self.assert_true(check_gradients_vector(forward, dic))

        return
    def test_grad_with_vec2(self):
        choices2 = [(3), (32), (12), (2)]
        choices = [(3, 3), (18, 32), (33, 12), (64, 2)]
        times = []

        def forward(a, b):
            c = sail.matmul(a, b)
            d = sail.sum(c)
            return d

        for ca, cb in zip(choices, choices2):
            arr1 = np.random.uniform(0, 1, (ca))
            arr2 = np.random.uniform(0, 1, (cb))

            dic = {
                "a": arr1,
                "b": arr2
            }

            self.assert_true(check_gradients_vector(forward, dic))

        return

class AddmmTest(UnitTest):

    @requires_grad_decorator
    def test_base(self, rq):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        for ca, cbs in zip(choices, choices_2):
            for cb in cbs:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca))
                arr2 = np.random.uniform(0, 1, (cb))
                arr3 = np.random.uniform(0, 1, (verif_shape))

                x1 = sail.Tensor(arr1, requires_grad=rq)
                x2 = sail.Tensor(arr2, requires_grad=rq)
                x3 = sail.Tensor(arr3, requires_grad=rq)

                x4 = sail.addmm(x1, x2, x3)
                arr4 = np.matmul(arr1, arr2) + arr3

                self.assert_eq(x4.shape, verif_shape)
                self.assert_eq_np_sail(arr4, x4, eps=1e-7)
                self.assert_eq(x4.requires_grad, rq)

        return

    def test_grad(self):
        choices = [(3, 3), (12, 18), (2, 33), (32, 64)]
        choices_2 = [[(3, 3), (3, 1), (3, 10)], [(18, 12), (18, 2)], [(33, 1), (33, 33)], [(64, 12)]]
        times = []
        def forward(a, b, c):
            d = sail.addmm(a, b, c)
            e = sail.sum(d)
            return e

        for ca, cbs in zip(choices, choices_2):
            for cb in cbs[:1]:
                verif_shape = (ca[0], cb[1])
                arr1 = np.random.uniform(0, 1, (ca))
                arr2 = np.random.uniform(0, 1, (cb))
                arr3 = np.random.uniform(0, 1, (verif_shape))

                dic = {
                    "a": arr1,
                    "b": arr2,
                    "c": arr3
                }

                self.assert_true(check_gradients_vector(forward, dic))

        return

    
    def test_error_base(self):
        a = sail.random.uniform(0, 1, (1))
        b = sail.random.uniform(1, 2, (1))
        c = sail.random.uniform(1, 2, (1))
        self.assert_throws(sail.addmm, (a, b, c), sail.SailError)

        a = sail.random.uniform(0, 1, (3, 4, 5))
        b = sail.random.uniform(1, 2, (5, 2))
        self.assert_throws(sail.addmm, (a, b, c), sail.SailError)


        a = sail.random.uniform(0, 1, (3, 5))
        b = sail.random.uniform(1, 2, (3, 5))
        self.assert_throws(sail.addmm, (a, b, c), sail.SailError)
        

class TensordotTest(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        choices = [
            {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': 2, 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': ([1, 2], [0, 1]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (4, 2, 3), 'b_shape': (3, 5, 2), 'axes': ([2, 1], [0, 2]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (2, 4, 3), 'b_shape': (5, 3, 2), 'axes': ([2, 0], [1, 2]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (2, 3, 4), 'b_shape': (5, 2, 3), 'axes': ([1, 0], [2, 1]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (3, 2, 4), 'b_shape': (2, 5, 3), 'axes': ([0, 1], [2, 0]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (3, 4, 2), 'b_shape': (2, 3, 5), 'axes': ([0, 2], [1, 0]), 'gc_shape': (4, 5)},  # NOQA
            {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': 1, 'gc_shape': (3, 4, 5, 6)},  # NOQA
            {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': ([2], [0]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
            {'a_shape': (3, 2, 4), 'b_shape': (5, 2, 6), 'axes': ([1], [1]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
            {'a_shape': (2, 3, 4), 'b_shape': (5, 6, 2), 'axes': ([0], [2]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
            {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': 2, 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': ([2, 3], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (4, 5, 2, 3), 'b_shape': (3, 6, 2), 'axes': ([3, 2], [0, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (4, 2, 5, 3), 'b_shape': (6, 3, 2), 'axes': ([3, 1], [1, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (2, 4, 5, 3), 'b_shape': (6, 2, 3), 'axes': ([3, 0], [2, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (2, 4, 3, 5), 'b_shape': (2, 6, 3), 'axes': ([2, 0], [2, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (2, 3, 4, 5), 'b_shape': (2, 3, 6), 'axes': ([1, 0], [1, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (3, 2, 4, 5), 'b_shape': (3, 2, 6), 'axes': ([0, 1], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
            {'a_shape': (3, 2, 5, 4), 'b_shape': (3, 6, 2), 'axes': ([0, 1], [0, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
            {'a_shape': (3, 5, 2, 4), 'b_shape': (6, 3, 2), 'axes': ([0, 2], [1, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
            {'a_shape': (5, 3, 2, 4), 'b_shape': (6, 2, 3), 'axes': ([1, 2], [2, 1]), 'gc_shape': (5, 4, 6)},  # NOQA
            {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': 3, 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': ([1, 2, 3], [0, 1, 2]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (5, 4, 2, 3), 'b_shape': (4, 3, 6, 2), 'axes': ([1, 3, 2], [0, 1, 3]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (5, 2, 4, 3), 'b_shape': (4, 6, 3, 2), 'axes': ([2, 3, 1], [0, 2, 3]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (2, 5, 4, 3), 'b_shape': (4, 6, 2, 3), 'axes': ([2, 3, 0], [0, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (2, 5, 3, 4), 'b_shape': (6, 4, 2, 3), 'axes': ([3, 2, 0], [1, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (2, 3, 5, 4), 'b_shape': (6, 2, 4, 3), 'axes': ([3, 1, 0], [2, 3, 1]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (3, 2, 5, 4), 'b_shape': (6, 2, 3, 4), 'axes': ([3, 0, 1], [3, 2, 1]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (3, 2, 4, 5), 'b_shape': (2, 6, 3, 4), 'axes': ([2, 0, 1], [3, 2, 0]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (3, 4, 2, 5), 'b_shape': (2, 3, 6, 4), 'axes': ([1, 0, 2], [3, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (4, 3, 2, 5), 'b_shape': (2, 3, 4, 6), 'axes': ([0, 1, 2], [2, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (4, 3, 5, 2), 'b_shape': (3, 2, 4, 6), 'axes': ([0, 1, 3], [2, 0, 1]), 'gc_shape': (5, 6)},  # NOQA
            {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 4, 2, 6), 'axes': ([0, 2, 3], [1, 0, 2]), 'gc_shape': (5, 6)},  # NOQA
        ]

        for c in choices:
            arr1 = np.random.uniform(0, 1, c["a_shape"])
            arr2 = np.random.uniform(0, 1, c["b_shape"])

            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)

            arr3 = np.tensordot(arr1, arr2, axes=c["axes"])
            x3 = sail.tensordot(x1, x2, axes=c["axes"])

            self.assert_eq_np_sail(arr3, x3, eps=1e-7)
