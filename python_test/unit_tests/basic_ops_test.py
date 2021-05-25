
import sail, sys, random, os, gc, time

import numpy as np

from ..test_utils import *

elementwise_options = [(12), (512, 128), (3, 14, 2), (8, 12, 12, 12), (3, 1, 5, 6), (13, 14)]

def test_create():

    choices = elementwise_options
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        x1 = sail.Tensor(arr1, requires_grad=False)

        if (not assert_eq_np_sail(arr1, x1)):
            return False

    return True

def test_add():
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

        assert_eq_np_sail(arr3, x3)

    log_time(np.mean(times), "ADD")

    return True

def test_sub():
    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        x1 = sail.Tensor(arr1, requires_grad=False)
        x2 = sail.Tensor(arr2, requires_grad=False)
        
        t = time.time()
        x3 = x1 - x2 
        times.append(time.time() - t)
        arr3 = arr1 - arr2 

        assert_eq_np_sail(arr3, x3)

    log_time(np.mean(times), "SUBTRACT")

    return True

def test_mult():
    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        x1 = sail.Tensor(arr1, requires_grad=False)
        x2 = sail.Tensor(arr2, requires_grad=False)
        
        t = time.time()
        x3 = x1 * x2 
        times.append(time.time() - t)
        arr3 = arr1 * arr2 

        assert_eq_np_sail(arr3, x3)

    log_time(np.mean(times), "MULTIPLY")

    return True

def test_divide():
    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        x1 = sail.Tensor(arr1, requires_grad=False)
        x2 = sail.Tensor(arr2, requires_grad=False)
        
        t = time.time()
        x3 = x1 / x2 
        times.append(time.time() - t)
        arr3 = arr1 / arr2 

        assert_eq_np_sail(arr3, x3)

    log_time(np.mean(times), "DIVIDE")

    return True

    