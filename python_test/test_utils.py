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

def to_significant(a, significant=7):
    mult = 10 ** significant
    div = 1/mult 

    a = np.floor(a * mult) * div 
    return a 

def assert_approx_equal(a, b, significant=7):
    mult = 10 ** significant
    a = np.floor(a * mult)
    b = np.floor(b * mult)
    assert a == b

def assert_eq(a, b):
    assert a == b

def assert_true(condition):
    assert condition

def log_time(time, function):
    return print ("TEST | %s mean execution time: %s" % (function, time))

def log_complete(function):
    return print ("TEST | %s complete" % (function))