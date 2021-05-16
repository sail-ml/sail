import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

def assert_eq_np_sail(np_arr, sail_arr):
    sail_np = sail_arr.numpy()
    # print (np.array_equal(np_arr, sail_np))
    return np.array_equal(np_arr, sail_np)

def assert_eq_np(np_arr, np_sail_arr):
    return np.array_equal(np_arr, np_sail_arr)

def report_time(time, function):
    return "%s mean execution time: %s" % (function, time)