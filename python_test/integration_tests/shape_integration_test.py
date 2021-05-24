import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

from ..test_utils import *


def test_shape_integration_1():
    arr1 = np.random.uniform(0, 1, (12, 32, 4))

    x1 = sail.Tensor(arr1, requires_grad=False)

    arr1 = np.transpose(arr1, (2, 0, 1))
    arr1 = np.reshape(arr1, (48, 32))
    arr1 = np.expand_dims(arr1, 0)
    arr1 = np.transpose(arr1, (1, 2, 0))

    x1 = sail.transpose(x1, (2, 0, 1))
    x1 = sail.reshape(x1, (48, 32))
    x1 = sail.expand_dims(x1, 0)
    x1 = sail.transpose(x1, (1, 2, 0))

    assert_eq_np_sail(arr1, x1)

    log_complete("SHAPE INTEGRATION 1")

def test_shape_integration_2():
    arr1 = np.random.uniform(0, 1, (12))

    x1 = sail.Tensor(arr1, requires_grad=False)

    arr1 = np.expand_dims(arr1, 0)
    arr1 = np.expand_dims(arr1, 0)
    arr1 = np.transpose(arr1, (0, 2, 1))
    arr1 = np.squeeze(arr1, 0)

    x1 = sail.expand_dims(x1, 0)
    x1 = sail.expand_dims(x1, 0)
    x1 = sail.transpose(x1, (0, 2, 1))
    x1 = sail.squeeze(x1, 0)

    assert_eq_np_sail(arr1, x1)

    log_complete("SHAPE INTEGRATION 2")

