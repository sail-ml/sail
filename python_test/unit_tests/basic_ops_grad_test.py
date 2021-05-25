
import sail, sys, random, os, gc, time

import numpy as np
from ..test_utils import *


def numel(x):
    s = x.shape 
    o = 1
    for i in s:
        o *= i 
    return o

def dictionary_to_vector(dictionary):
    return np.concatenate([dictionary[i].ravel() for i in dictionary])

def vector_to_dictionary(vector, dictionary):
    nd = {}
    start = 0
    for d in dictionary:
        nd[d] = None
        data = dictionary[d]
        end = start + int(numel(data))   
        nd[d] = vector[start:end].reshape(data.shape)
        start += int(numel(data))   
    return nd


dic = {"a": np.random.uniform(1, 2, (35)), 
       "b": np.random.uniform(1, 2, (35)),
       }

def forward(a, b):
    c = sail.multiply(a, b)
    d = sail.sum(c)
    return d


def check_gradients_vector(forward_fcn, param_dictionary):
    params = [sail.Tensor(param_dictionary[a], requires_grad=True) for a in param_dictionary]
    output = forward_fcn(*params)
    output.backward()
    grads_dic = {}
    for i, p in enumerate(param_dictionary):
        grads_dic[p] = params[i].grad.numpy() 

    parameters = dictionary_to_vector(param_dictionary)
    grads = dictionary_to_vector(grads_dic)
    num_params = len(parameters)

    j_plus = np.zeros(num_params)
    j_minus = np.zeros(num_params)
    grad_approx = np.zeros(num_params)

    eps = 1e-7
    for i in range(len(parameters)):
        # pass
        params_plus = np.copy(parameters)
        params_plus[i] += eps 
        params_plus_dict = vector_to_dictionary(params_plus, param_dictionary)
        params_plus_ = [sail.Tensor(params_plus_dict[a]) for a in params_plus_dict]
        
        z = forward_fcn(*params_plus_)
        j_plus[i] = z.numpy()

        params_minus = np.copy(parameters)
        params_minus[i] -= eps 
        params_minus_dict = vector_to_dictionary(params_minus, param_dictionary)
        params_minus_ = [sail.Tensor(params_minus_dict[a]) for a in params_minus_dict]
        
        z = forward_fcn(*params_minus_)
        j_minus[i] = z.numpy()

        grad_approx[i] = (j_plus[i] - j_minus[i])/(2 * eps)
        grad_approx[i] = to_significant(grad_approx[i], significant=7)

    grads = to_significant(grads, significant=7)

    num = np.linalg.norm(grad_approx - grads)
    denom = np.linalg.norm(grad_approx) + np.linalg.norm(grads)
    diff = num/denom
    return diff 

# diff = check_gradients_vector(forward, dic)
# if diff < 1e-6:
#     print ("gradient is correct, %s" % diff)
# else:
#     print ("gradient is incorrect, %s" % diff)

# dic2 = vector_to_dictionary(vector, dic)

elementwise_options = [(12), (3, 14, 2), (8, 12, 12), (3, 1, 5, 6), (13, 14)]

def test_add_grad():

    def forward(a, b):
        c = sail.add(a, b)
        d = sail.sum(c)
        return d

    choices = elementwise_options
    # choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        dic = {
            "a": arr1,
            "b": arr2
        }

        diff = check_gradients_vector(forward, dic)
        assert diff < 1e-6

    log_complete("ADD GRAD")
    return True
    
def test_sub_grad():

    def forward(a, b):
        c = sail.subtract(a, b)
        d = sail.sum(c)
        return d

    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        dic = {
            "a": arr1,
            "b": arr2
        }

        diff = check_gradients_vector(forward, dic)

        assert diff < 1e-6

    log_complete("SUBTRACT GRAD")

    return True

def test_mult_grad():

    def forward(a, b):
        c = sail.multiply(a, b)
        d = sail.sum(c)
        return d

    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        dic = {
            "a": arr1,
            "b": arr2
        }

        diff = check_gradients_vector(forward, dic)

        assert diff < 1e-6

    log_complete("MULTIPLY GRAD")

    return True

def test_divide_grad():

    def forward(a, b):
        c = sail.divide(a, b)
        d = sail.sum(c)
        return d

    choices = elementwise_options
    times = []
    for c in choices:
        arr1 = np.random.uniform(0, 1, (c))
        arr2 = np.random.uniform(0, 1, (c))

        dic = {
            "a": arr1,
            "b": arr2
        }

        diff = check_gradients_vector(forward, dic)

        assert diff < 1e-5

    log_complete("DIVIDE GRAD")

    return True

def test_matmul_grad():

    def forward(a, b):
        c = sail.matmul(a, b)
        d = sail.sum(c)
        return d

    choices_a = [(12, 12), (3, 4), (5, 12), (100, 30)]
    choices_b = [(12, 3), (4, 18), (12, 5), (30, 25)]
    times = []
    for ca, cb in zip(choices_a, choices_b):
        arr1 = np.random.uniform(0, 1, (ca))
        arr2 = np.random.uniform(0, 1, (cb))

        dic = {
            "a": arr1,
            "b": arr2
        }

        diff = check_gradients_vector(forward, dic)

        assert diff < 1e-6

    log_complete("MATMUL GRAD")

    return True


