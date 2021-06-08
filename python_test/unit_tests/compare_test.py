
import sail, sys, random, os, gc, time, torch

import numpy as np

from ..test_utils import *


def test_linear_torch():

    linear_sail = sail.modules.Linear(32, 2)
    linear_torch = torch.nn.Linear(32, 2)

    weights = linear_sail.weights.numpy()
    biases = linear_sail.bias.numpy()


    with torch.no_grad():
        linear_torch.weight = torch.nn.Parameter(torch.from_numpy(weights.T))
        linear_torch.bias = torch.nn.Parameter(torch.from_numpy(biases))


    inputs = np.random.uniform(0, 1, (12, 32)).astype(np.float32)
    input2 = np.random.uniform(0, 1, (12, 32)).astype(np.float32)

    inputs_sail = sail.Tensor(inputs, requires_grad=True)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    out_sail = linear_sail(inputs_sail)
    out_torch = linear_torch(inputs_torch)

    assert_eq_margin(out_sail.numpy(), out_torch.detach().numpy())

    torch.sum(out_torch).backward()
    sail.sum(out_sail).backward()

    print (linear_sail.weights.grad)

    assert_eq_margin(linear_sail.weights.grad.numpy().T, linear_torch.weight.grad.numpy(), 1e-4)
    # assert_eq_margin(inputs_sail.grad.numpy(), inputs_torch.grad.numpy(), 1e-4)
