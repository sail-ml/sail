import unittest
import sail, sys, random, os, gc, time
import psutil
import numpy as np

def assert_eq_np_sail(np_arr, sail_arr):
    sail_np = sail_arr.numpy()
    # print (np.array_equal(np_arr, sail_np))
    assert np.array_equal(np_arr, sail_np) == True

def assert_eq_np(np_arr, np_sail_arr):
    assert np.array_equal(np_arr, np_sail_arr) == True

def log_time(time, function):
    return print ("TEST | %s mean execution time: %s" % (function, time))

def log_complete(function):
    return print ("TEST | %s complete" % (function))