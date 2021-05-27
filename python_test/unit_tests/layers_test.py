
import sail, sys, random, os, gc, time

import numpy as np

from ..test_utils import *

def test_linear():

    linear = sail.modules.Linear(10, 2, use_bias=True)

    inputs = sail.random.uniform(1, 2, (4, 10))


    def _forward(inputs, weights, bias):
        return np.matmul(inputs, weights) + bias 

    np_weights = linear.weights.numpy()
    np_bias = linear.bias.numpy()

    res = _forward(inputs.numpy(), np_weights, np_bias)
    res_s = linear(inputs)

    assert_eq_np_sail_margin(res, res_s, margin=1e-7)

    log_complete("LINEAR LAYER")
    