import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

def assert_eq_np_sail(np_arr, sail_arr):
    sail_np = sail_arr.numpy()
    # print (np.array_equal(np_arr, sail_np))
    return np.array_equal(np_arr, sail_np)

def report_time(time, function):
    return "%s mean execution time: %s" % (function, time)

def create_test():

    choices = list(range(1, 128)) + list(range(256, 256**2, 256))
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        x1 = sail.Tensor(arr1, requires_grad=False)

        if (not assert_eq_np_sail(arr1, x1)):
            return False

    return True

def add_test():
    choices = list(range(1, 128)) + list(range(256, 256**2, 256))
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        x1 = sail.Tensor(arr1, requires_grad=False)
        x2 = sail.Tensor(arr2, requires_grad=False)
        
        t = time.time()
        x3 = x1 + x2 
        times.append(time.time() - t)
        arr3 = arr1 + arr2 

        if (not assert_eq_np_sail(arr3, x3)):
            return False

    print (report_time(np.mean(times), "ADD"))

    return True

def sub_test():
    choices = list(range(1, 128)) + list(range(256, 256**2, 256))
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

        if (not assert_eq_np_sail(arr3, x3)):
            return False

    print (report_time(np.mean(times), "SUBTRACT"))

    return True

def mult_test():
    choices = list(range(1, 128)) + list(range(256, 256**2, 256))
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

        if (not assert_eq_np_sail(arr3, x3)):
            return False

    print (report_time(np.mean(times), "MULTIPLY"))

    return True

def divide_test():
    choices = list(range(1, 128)) + list(range(256, 256**2, 256))
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

        if (not assert_eq_np_sail(arr3, x3)):
            return False

    print (report_time(np.mean(times), "DIVIDE"))

    return True

    