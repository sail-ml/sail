import sail, sys, random
import numpy as np
import cupy as cp
import ctypes
import pycuda.driver as cuda
from cupy.cuda import memory
import time, torch

shape = 16000

linear_test_shapes = list(range(1, 128, 4)) + list(range(256, 256**2, 256)) + [25]

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


        assert(np.sum(op(arr1, arr2)) == np.sum(op(x1, x2).numpy()))

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
# # # # benchmark_shapes(nd_test_shape, sub)

# print ("\nMUL")
# benchmark_shapes(linear_test_shapes, mul)
# # # # benchmark_shapes(nd_test_shape, mul)

# print ("\nDIV")
# benchmark_shapes(linear_test_shapes, truediv)
# benchmark_shapes(nd_test_shape, truediv)

# arr2 = np.random.uniform(0, 1, (32000))#, 32))
# arr1 = np.random.uniform(0, 1, (32000))#, 32))

# x1 = sail.Tensor(arr1)

# print (sail.mean(x1).numpy())

# # x3 = x1 + x1
# # x3 = sail.multiply(x1, 2.0)
# x3 = x1 * x1
# x3 = x1 * 2.0
# print (x1.numpy())
# print (x3.numpy())

# print (np.sum(arr1))
# print (sail.sum(x1).numpy())

# print (sail.add.__doc__)


arr1 = np.random.uniform(0, 1, (5, 20, 2)).astype(np.float64)
arr2 = np.random.uniform(0, 1, (5, 20, 2)).astype(np.float64)

# axis = 1
# print (arr2.strides)
# x1 = sail.Tensor(arr2)
# print (np.sum(arr2, axis))
# print ("EXP STRIDES", np.sum(arr2, axis).strides)
# print ("EXP SHAPE", np.sum(arr2, axis).shape)
# print ("STRIDE CALC", 8 * np.r_[1, np.cumprod(np.sum(arr2, axis).shape[::-1][:-1])][::-1])
# # x2 = x1[0]
# # print (arr2[0].shape)
# # print (arr2[0])
# # print (x2.numpy().shape)
# # print (x2.numpy())
# # print (x1[0].numpy())
# # s1 = sail.sum(x1)
# s0 = sail.sum(x1, axis)
# # s2 = sail.sum(x1, 1)
# # s3 = sail.sum(x1, 2)

# print (s0.numpy())
c = 0
for z in range(10):
    for i in range(2, 5):
        for axis in range(i):
            # print (c, i, axis)
            shape = [random.randint(0, 100) for _ in range(i)]
            print (shape)
            arr1 = np.random.uniform(0, 1, shape).astype(np.float64)

            x1 = sail.Tensor(arr1)

            sum_np = np.sum(arr1, axis=axis)
            sum_sail = sail.sum(x1, axis=axis).numpy()

            # assert(sum_np.all() == sum_sail.all())
            c += 1

print ("DONE")

# print (np.sum(arr2), np.sum(arr2).strides)
# print (np_v.numpy(), np_v.numpy().strides)
arr1 = np.random.uniform(0, 100, (4, 2)).astype(np.float32)
arr2 = np.random.uniform(0, 100, (2, 4)).astype(np.float32)

print (np.matmul(arr1, arr2))

x1 = sail.Tensor(arr1)
x2 = sail.Tensor(arr2)

y = sail.matmul(x1, x2)
print (y.numpy())


# print (x1.numpy())
# print (sail.int32)
# print (np.int32)
# x2 = x1.astype(sail.int32)
# print (x2.numpy())
# x3 = x2.astype(sail.float64)
# print (x3.numpy())

# print (arr1 // arr2)
# t = time.time()
# for i in range(100):
#     arr1 / 3 
# print ((time.time() - t)/100)
# t = time.time()
# for i in range(100):
#     arr1 / 3.0 
# print ((time.time() - t)/100)

# x1 = sail.Tensor(arr1)
# # x2 = sail.Tensor(arr2)
# t = time.time()
# for _ in range(100):
#     arr1.astype(np.int32)
# print ((time.time() - t)/100)
# y = sail.cast_int32(x1)
# t = time.time()
# for _ in range(100):
#     y-y
# print ((time.time() - t)/100)
# print (y.numpy())
# # print((y + y).numpy())
# # print((y - y).numpy())
# # print((y * y).numpy())
# # print((y - y).numpy())
# print((y *y).numpy())
# x2 = x1 / 3.0
# print (x2.numpy())

# print (x1.numpy())

# x3 = sail.add(x1, x2)#x1 + x2
# print (x3.numpy())

# print (x1.numpy())
# print (sail.cast_int32(x1).numpy())
# print (x1.numpy())

# x3 = x2 + x1
# print (x3.numpy())
# x2 = sail.Tensor(arr2)
# # print (x1.numpy())
# x3 = x1 / x2
# print (x3.numpy())
# print (dir(arr1))
# print (arr1.ctypes)

