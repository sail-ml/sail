import sail, sys, random
import numpy as np
import cupy as cp
import ctypes
import pycuda.driver as cuda
from cupy.cuda import memory
import time, torch

shape = 16000

linear_test_shapes = list(range(1, 128, 4)) + list(range(256, int(256**2), 256))
# linear_test_shapes =  list(range(256, int(256**1.2), 256)) + [25] #list(range(4, 128, 4)) +

nd_test_shape = [(16, 32), (32, 128), (256, 4), (784, 128), (4096, 4096)]

def add(a, b):
    return a+b
def mul(a, b):
    return a*b
def sub(a, b):
    return a-b
def truediv(a, b):
    return a/b

def benchmark_binary(arr1, arr2, op, iters):
    # op(arr1, arr2)
    t = time.time()
    for i in range(iters):
        op(arr1, arr2)
    t = (time.time() - t) / iters
    # time.sleep(1)
    return t

def benchmark_shapes(shapes, op, verbose=False, grad=False):
    faster = {"SAIL": 0, "NUMPY": 0, "FAIL":[]}
    sails = []
    numpys = []
    np_time = 0
    sail_time = 0
    for s in shapes:
        # print (s)
        # for i in range(10000):
        arr1 = np.random.uniform(0, 1, s)
        arr2 = np.random.uniform(0, 1, s)

        x1 = sail.Tensor(arr1, requires_grad=grad)
        x2 = sail.Tensor(arr2, requires_grad=grad)

        assert(np.sum(op(arr1, arr2)) == np.sum(op(x1, x2).numpy()))
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
#     #     print ("FAILED ON: %s" % faster["FAIL"])
# print ("ADD")

# benchmark_shapes(linear_test_shapes, add, grad=False)
# exit()
# # benchmark_shapes(nd_test_shape, add)

# print ("\nSUB")
# benchmark_shapes(linear_test_shapes, sub, grad=False)
# # # # # # benchmark_shapes(nd_test_shape, sub)

# print ("\nMUL")
# benchmark_shapes(linear_test_shapes, mul, grad=False)
# # # # # # benchmark_shapes(nd_test_shape, mul)

# print ("\nDIV")
# benchmark_shapes(linear_test_shapes, truediv, grad=False)
# # benchmark_shapes(nd_test_shape, truediv)

# # arr2 = np.random.uniform(0, 1, (32000))#, 32))
# arr1 = np.random.uniform(0, 1, (3))#, 32))
# arr2 = np.random.uniform(0, 1, (3, 3, 2, 3))#, 32))

# print (arr1.strides)

# x1 = sail.Tensor(arr1, requires_grad=True)
# x2 = sail.broadcast_to(x1, (3, 3))

# print (x2.shape)
# print (x2.numpy())
# x2 = sail.Tensor(arr2, requires_grad=True)

# # print (sail.mean(x1).numpy())

# x3 = x1 + x2
# print (x3.shape)
# print (" ")
# print (arr1 + arr2)
# # x3 = sail.multiply(x1, 2.0)
# x3 = x1 * x1
# x3 = x1 * 2.0
# print (x1.numpy())
# print (x3.numpy())

# print (np.sum(arr1))
# print (sail.sum(x1).numpy())

# print (sail.add.__doc__)
# x = np.random.uniform(0, 1, (1, 10))
# x = np.broadcast_to(x, (5, 10))
# print (x)

# arr1 = np.random.uniform(0, 1, (5, 20, 2)).astype(np.float64)
# arr2 = np.random.uniform(0, 1, (5, 20, 2)).astype(np.float64)

# for z in range(256, 256*4, 8):
#     print (z)
arr1 = np.random.uniform(1, 2, (2,4,3)).astype(np.float64)#, 32))
arr2 = np.random.uniform(1, 2, (5,3,2)).astype(np.float64)#, 32))

# # for i in range(1000000000):
x1 = sail.Tensor(arr1, requires_grad=False)
x2 = sail.Tensor(arr2, requires_grad=False)

print (x1)
print (x2)

print (np.tensordot(arr1, arr2, axes=([2, 0], [1, 2])))
print (sail.tensordot(x1, x2, axes=([2, 0], [1, 2])))

exit()
