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

def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes.
    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.
    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".
    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.
    Returns
    -------
    output : ndarray
        The tensor dot product of the input.
    See Also
    --------
    dot, einsum
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.
    Examples
    --------
    A "traditional" example:
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> # A slower but equivalent way of computing the same...
    >>> d = np.zeros((5,2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i,j] += a[k,n,i] * b[n,k,j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])
    An extended example taking advantage of the overloading of + and \\*:
    >>> a = np.array(range(1, 9))
    >>> a.shape = (2, 2, 2)
    >>> A = np.array(('a', 'b', 'c', 'd'), dtype=object)
    >>> A.shape = (2, 2)
    >>> a; A
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    array([['a', 'b'],
           ['c', 'd']], dtype=object)
    >>> np.tensordot(a, A) # third argument default is 2 for double-contraction
    array(['abbcccdddd', 'aaaaabbbbbbcccccccdddddddd'], dtype=object)
    >>> np.tensordot(a, A, 1)
    array([[['acc', 'bdd'],
            ['aaacccc', 'bbbdddd']],
           [['aaaaacccccc', 'bbbbbdddddd'],
            ['aaaaaaacccccccc', 'bbbbbbbdddddddd']]], dtype=object)
    >>> np.tensordot(a, A, 0) # tensor product (result too long to incl.)
    array([[[[['a', 'b'],
              ['c', 'd']],
              ...
    >>> np.tensordot(a, A, (0, 1))
    array([[['abbbbb', 'cddddd'],
            ['aabbbbbb', 'ccdddddd']],
           [['aaabbbbbbb', 'cccddddddd'],
            ['aaaabbbbbbbb', 'ccccdddddddd']]], dtype=object)
    >>> np.tensordot(a, A, (2, 1))
    array([[['abb', 'cdd'],
            ['aaabbbb', 'cccdddd']],
           [['aaaaabbbbbb', 'cccccdddddd'],
            ['aaaaaaabbbbbbbb', 'cccccccdddddddd']]], dtype=object)
    >>> np.tensordot(a, A, ((0, 1), (0, 1)))
    array(['abbbcccccddddddd', 'aabbbbccccccdddddddd'], dtype=object)
    >>> np.tensordot(a, A, ((2, 1), (1, 0)))
    array(['acccbbdddd', 'aaaaacccccccbbbbbbdddddddd'], dtype=object)
    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = np.asarray(a), np.asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    print (axes_a, axes_b)

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    print (newshape_a)
    print (newshape_b)
    print (newaxes_a)
    print (newaxes_b)
    print (a.transpose(newaxes_a).shape)
    print (b.transpose(newaxes_b).shape)

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)

    # print (at)
    print (a.transpose(newaxes_a))
    print (a.transpose(newaxes_a).reshape(newshape_a))
    res = np.dot(at, bt)
    return res.reshape(olda + oldb)
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
arr1 = np.random.uniform(1, 2, (2,2,2)).astype(np.float64)#, 32))
arr2 = np.random.uniform(1, 2, (2,2,2)).astype(np.float64)#, 32))

# # for i in range(1000000000):
x1 = sail.Tensor(arr1, requires_grad=False)
x2 = sail.Tensor(arr2, requires_grad=False)

print (x1)
print (x2)
# print (sail.transpose(x1))
# print (x1)
# print (sail.transpose(x1, (1, 0, 2)))
# print (np.tensordot(arr1, arr2))
# print (np.tensordot(arr1, arr2).shape)
# print (np.tensordot(arr1, arr2, axes=([0,1],[0,1])))
# print (np.tensordot(arr1, arr2, axes=([1,0],[0,1])).shape)

print (sail.tensordot(x1, x2, axes=((1, 0), (0, 1))))
# print (sail.tensordot(x1, x2, axes=((0, 1), (0, 1))))
# print (sail.tensordot(x1, x2))
# print (sail.tensordot(x1, x2, axes=((1, 0), (1, 0))))
# print (sail.tensordot(x1, x2, axes=((0, 1), (1, 0))))

print (np.tensordot(arr1, arr2, axes=((1,0),(0,1))))
print (tensordot(arr1, arr2, axes=((1,0),(0,1))))
exit()
# print (np.tensordot(arr1, arr2, axes=((0,1),(0,1))))
# print (sail.tensordot(x1, x2))#, axes=((1, 0), (0, 1))))
# print (sail.tensordot(x1, x2))
# print (x1)
arr1 = np.random.uniform(0, 1, (3, 4)).astype(np.float64)#, 32))
arr2 = np.random.uniform(0, 1, (4, 3)).astype(np.float64)#, 32))

# for i in range(1000000000):
x1 = sail.Tensor(arr1, requires_grad=False)
x2 = sail.Tensor(arr2, requires_grad=False)

# print (x1)
# print (sail.transpose(x1))
# print (x1)
# print (sail.transpose(x1, (1, 0, 2)))
print (np.dot(arr1, arr2))
print (np.dot(arr1, arr2).shape)

print (sail.matmul(x1, x2))
# print (x1)
# print (x2)

# x3 = sail.divide(x1, x2) 
# x4 = sail.sum(x3)

# x4.backward()

# print (" ")

# print (x4.grad)
# print (x3.grad)
# print (x2.grad)
# print (x1.grad)
# for i in range(100000000):
#     x1 = sail.Tensor(arr1, requires_grad=False)
#     x2 = sail.expand_dims(x1, 2)
#     # print (x3)
    # print (x4)
    # print (x5)
    # print (x1)
    # print (x2.numpy())
# print (x2.numpy())
# # print (x1)
# x2 = sail.Tensor(arr2, requires_grad=True)

# # for i in range(100):
# t = time.time()
# for i in range(100000):
#     x3 = sail.add(x1, x2)

# print (x3)
# print (arr1 + arr2)
    # print (x3)
    # print (x3)
    # x4 = sail.sum(x3)
    # x3.backward()
    # print ("back complete")
    # print (x4.grad)
    # print (x3.grad)
    # print (x2.grad)
    # print (x1.grad)
    # exit()
# print (x3)
# print (x3)
# error when not added here. IDK why
# print (x3.grad)
# print (x3.grad)
# print (x2.grad)
# print (x1.grad)



# print (sys.getrefcount(x3))
# print (x3)
# print (x3.grad)
# # x4 = sail.sum(x3)
# # print (x4)
# # print (x4)

# # print (x4)
# print ("z")
# x4.backward()
# print (x4.grad)
# print (x3.grad)
# print (x2.grad)
# print (x1.grad)
# print ("\n\n\n ")
# print (x4)
# print (x3)
# print (x2)
# print (x1)
# print (x4)
# # # # # # print (x1.get_grad())
# # # # # # print (x1)
# # # # # # print (x4)
# # # # # print (x3.grad)

# print (x1.grad)
# print (x1.grad.numpy())
# print (x4)
# print (x4.numpy())
# import gc
# for i in range(100000):
#     arr1 = np.random.uniform(0, 1, (32000)).astype(np.float64)#, 32))
#     # gc.collect()
#     x1 = sail.Tensor(arr1, requires_grad=True)
    
