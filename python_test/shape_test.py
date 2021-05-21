import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

from test_utils import *



def test_reshape():
    choices = [(10, 30), (35, 4), (12), (13, 21, 3), (8, 12, 4, 6)]
    reshapes = [[(30, 10), (3, 100), (300,)], [(4, 35)], [(2, 2, 3), (6, 2)], [(21, 3, 13), (39, 21)], [(6, 12, 8, 2, 2), (8, 144, 2)]]
    times = []
    i = 0

    for c, res in zip(choices, reshapes):
        for r in res:
            arr1 = np.random.uniform(0, 1, (c))

            x1 = sail.Tensor(arr1, requires_grad=False)
        
            arr1 = np.reshape(arr1, r)
            x1 = sail.reshape(x1, r)

            assert_eq_np_sail(arr1, x1)

    log_complete("RESHAPE")
    return True

def test_expand_dims():
    choices = [(10, 30), (35, 4), (12), (13, 21, 3), (8, 12, 4, 6)]
    dims = [(0, 1, 2, -1), (0, 1, 2, -1), (0, 1, -1), (0, 1, 2, 3, -1), (0, 1, 2, 3, 4, -1)]
    times = []
    i = 0

    for c, dis in zip(choices, dims):
        for d in dis:
            arr1 = np.random.uniform(0, 1, (c))

            x1 = sail.Tensor(arr1, requires_grad=False)
        
            arr1 = np.expand_dims(arr1, d)
            x1 = sail.expand_dims(x1, d)

            assert_eq_np_sail(arr1, x1)
            assert_eq(arr1.shape, x1.shape)

    log_complete("EXPAND DIMS")
    return True

def test_squeeze():
    choices = [(10, 30, 1), (1, 2, 1), (12, 1, 8, 1)]
    dims = [(-1,), (0, 2, -1), (1, 3, -1)]
    times = []
    i = 0

    for c, dis in zip(choices, dims):
        for d in dis:
            arr1 = np.random.uniform(0, 1, (c))

            x1 = sail.Tensor(arr1, requires_grad=False)
        
            arr1 = np.squeeze(arr1, d)
            x1 = sail.squeeze(x1, d)

            assert_eq_np_sail(arr1, x1)
            assert_eq(arr1.shape, x1.shape)

    log_complete("SQUEEZE")
    return True