import sail, sys, random
import numpy as np
import cupy as cp
import ctypes
import pycuda.driver as cuda
from cupy.cuda import memory
import time, torch

shape = 16000

linear_test_shapes = list(range(0, 128, 4)) + list(range(256, 256**2, 256))

nd_test_shape = [(16, 32), (32, 128), (256, 4), (784, 128), (4096, 4096)]

def add(a, b):
    a+b
def mul(a, b):
    a*b
def sub(a, b):
    a-b
def truediv(a, b):
    a/b

def benchmark_binary(arr1, arr2, op, iters):
    # op(arr1, arr2)
    t = time.time()
    for i in range(iters):
        op(arr1, arr2)
    t = (time.time() - t) / iters
    # time.sleep(1)
    return t

def benchmark_shapes(shapes, op, verbose=False):
    faster = {"SAIL": 0, "NUMPY": 0, "FAIL":[]}
    sails = []
    numpys = []
    for s in shapes:
        arr1 = np.random.uniform(0, 1, s)
        arr2 = np.random.uniform(0, 1, s)

        x1 = sail.Tensor(arr1)
        x2 = sail.Tensor(arr2)


        np_time = benchmark_binary(arr1, arr2, op, 100)
        # time.sleep(0.05)
        sail_time = benchmark_binary(x1, x2, op, 100)

        # time.sleep(0.05)

        sails.append(sail_time)
        numpys.append(np_time)

        if verbose:
            print ("SAIL TIME: %s" % sail_time)
            print ("NP TIME: %s" % np_time)
            if np_time > sail_time:
                print ("SAIL FASTER THAN NUMPY FOR %s elements" % s)
            else:
                print ("NUMPY FASTER THAN SAIL FOR %s elements" % s)

            print (" ")
        if sail_time < np_time:
            faster["SAIL"] += 1
        else:
            faster["NUMPY"] += 1
            faster["FAIL"].append(s)

    print ("SAIL FASTER FOR %s/%s" % (faster["SAIL"], len(shapes)))
    print (np.mean(sails), np.mean(numpys))
    # if faster["FAIL"] != []:
    #     print ("FAILED ON: %s" % faster["FAIL"])
# print ("ADD")
# benchmark_shapes(linear_test_shapes, add)
# # benchmark_shapes(nd_test_shape, add)

# print ("\nSUB")
# benchmark_shapes(linear_test_shapes, sub)
# # # benchmark_shapes(nd_test_shape, sub)

# print ("\nMUL")
# benchmark_shapes(linear_test_shapes, mul)
# # # benchmark_shapes(nd_test_shape, mul)

# print ("\nDIV")
# benchmark_shapes(linear_test_shapes, truediv)
# benchmark_shapes(nd_test_shape, truediv)

arr2 = np.random.uniform(0, 1, (32000))#, 32))
arr1 = np.random.uniform(0, 1, (32000))#, 32))

x1 = sail.Tensor(arr1)

# x3 = x1 + x1
# x3 = sail.multiply(x1, 2.0)
x3 = x1 * 2.0
print (x1.numpy())
print (x3.numpy())

print (np.sum(arr1))
print (sail.sum(x1).numpy())

print (sail.add.__doc__)
