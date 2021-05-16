import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

from test_utils import *



def test_reshape():
    choices = [(10, 30)]#, (35, 4), (12), (13, 21, 3), (8, 12, 4, 6)]
    reshapes = [[(30, 10),  (3, 100), (300)]]#, (3, 100), (300)]]#, [(4, 35)], [(21, 3, 13), (39, 21)], [(6, 12, 8, 2, 2), (8, 144, 2)]]
    times = []
    i = 0
    for c, res in zip(choices, reshapes):
        for r in res:
            arr1 = np.random.uniform(0, 1, (c))

            x1 = sail.Tensor(arr1, requires_grad=False)
        
            arr1 = np.reshape(arr1, r)
            x1 = sail.reshape(x1, r)

            if (not assert_eq_np(np.sum(arr1, 0), np.sum(x1.numpy(), 0))):
                return False
