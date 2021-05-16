import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

from test_utils import *

def test_matmul():
    choices_1 = [(10, 20), (300, 500), (1, 2), (2, 1), (8, 8)]
    choices_2 = [[(20, 10), (20, 3)], [(500, 150)], [(2, 2), (2, 1)], [(1, 2)], [(8, 1), (8, 4)]]
    times = []
    for c, ba in zip(choices_1, choices_2):
        for b in ba:
            arr1 = np.random.uniform(0, 1, c)
            arr2 = np.random.uniform(0, 1, b)

            x1 = sail.Tensor(arr1, requires_grad=False)
            x2 = sail.Tensor(arr2, requires_grad=False)
            
            t = time.time()
            x3 = sail.matmul(x1, x2)
            times.append(time.time() - t)
            arr3 = np.matmul(arr1, arr2)

            err = abs(np.sum(arr3) - np.sum(x3.numpy()))
            assert_true(err < 1e-8) # there are slight numerical differences

    log_time(np.mean(times), "MATMUL")

    return True

    