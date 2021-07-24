from test_case import *
import numpy as np
import sail
import time
import unittest, random

shapes = [(15, 10), (20, 30), (40, 5, 12, 3), (8000, 320), (1000000)]
minmax = [[(-1, 1.5), (5, 10), (2, 8)], [(13, 15), (-1.33, 2)], [(1, 2)], [(2, 3), (-1.2434, 0.123)], [(0, 1)]]
decs = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2]

class RandomUniform(UnitTest):

    def test_base(self):
        
        for sh, mm, d in zip(shapes, minmax, decs):
            for _mm in mm:
                x1 = sail.random.uniform(_mm[0], _mm[1], sh)
                arr1 = x1.numpy()

                if (isinstance(sh, int)):
                    sh = (sh, )
                self.assert_eq(x1.shape, sh)
                self.assert_lte(np.max(arr1), _mm[1])
                self.assert_gte(np.min(arr1), _mm[0])

                if (np.size(arr1) > 10000):
                    self.assert_eq_np(np.mean(arr1), np.array([sum(_mm)/2]), eps=d)

        return

class RandomUniformLike(UnitTest):

    def test_base(self):
        
        for sh, mm, d in zip(shapes, minmax, decs):
            for _mm in mm:
                x1 = sail.random.uniform(_mm[0], _mm[1], sh)
                x2 = sail.random.uniform_like(x1, _mm[0], _mm[1])
                arr1 = x2.numpy()
                arr2 = x1.numpy()

                if (isinstance(sh, int)):
                    sh = (sh, )
                self.assert_eq(x2.shape, sh)
                self.assert_lte(np.max(arr1), _mm[1])
                self.assert_gte(np.min(arr1), _mm[0])

                if (np.size(arr1) > 10000):
                    self.assert_eq_np(np.mean(arr1), np.array([sum(_mm)/2]), eps=d)

                self.assert_neq_np(arr1, arr2)

        return

class RandomNormal(UnitTest):

    def test_base(self):
        
        for sh, mm, d in zip(shapes, minmax, decs):
            for _mm in mm:
                x1 = sail.random.normal(_mm[0], _mm[1], sh)
                arr1 = x1.numpy()

                if (isinstance(sh, int)):
                    sh = (sh, )
                self.assert_eq(x1.shape, sh)

                if (np.size(arr1) > 1000000):
                    self.assert_eq_np(np.mean(arr1), np.array([_mm[0]]), eps=d)
                    self.assert_eq_np(np.std(arr1), np.array([_mm[1]]), eps=d)

        return

    def test_error(self):
        self.assert_throws(sail.random.uniform, (0, -1, (10, 20)), sail.SailError)

        return

class RandomNormalLike(UnitTest):

    def test_base(self):
        
        for sh, mm, d in zip(shapes, minmax, decs):
            for _mm in mm:
                x1 = sail.random.normal(_mm[0], _mm[1], sh)
                x2 = sail.random.normal_like(x1, _mm[0], _mm[1])
                arr1 = x2.numpy()
                arr2 = x1.numpy()

                if (isinstance(sh, int)):
                    sh = (sh, )
                self.assert_eq(x2.shape, sh)
               

                if (np.size(arr1) > 100000):
                    self.assert_eq_np(np.mean(arr1), np.array([_mm[0]]), eps=d)
                    self.assert_eq_np(np.std(arr1), np.array([_mm[1]]), eps=d)

                self.assert_neq_np(arr1, arr2)

        return
