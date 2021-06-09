
# import sail, sys, random, os, gc, time

# import numpy as np

# from ..test_utils import *

# def test_linear():

#     def _forward(inputs, weights, bias):
#         return np.matmul(inputs, weights) + bias 

#     # for i in range(10):
#     inp = random.randint(64, 256)
#     out = random.randint(64, 256)
#     batch = random.randint(64, 256)
#     linear = sail.modules.Linear(inp, out, use_bias=True)

#     inputs = sail.random.uniform(1, 2, (batch, inp))


#     np_weights = linear.weights.numpy()
#     np_bias = linear.bias.numpy()

#     res = _forward(inputs.numpy(), np_weights, np_bias)
#     res_s = linear(inputs)
#     # exit()

#     assert_approx_equal(np.sum(res), np.sum(res_s.numpy()), significant=5)

#     log_complete("LINEAR LAYER")
    