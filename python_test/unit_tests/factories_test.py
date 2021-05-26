
import sail, sys, random, os, gc, time

import numpy as np

from ..test_utils import *


def test_random_uniform():

    shapes = [(2, 5), (20, 30), (40, 5, 12, 3), (800, 12)]
    minmax = [[(-1, 1), (10, 5), (2, 8)], [(13, 15), (-10, -4)], [(1, 10)], [(0, 0)]]

    for s, _mm in zip(shapes, minmax):
        for mm in _mm:
            x1 = sail.random.uniform(mm[0], mm[1], s)
            diff = abs(np.mean(x1.numpy())) - abs(np.mean(mm))
            diff = abs(diff)
            assert (diff - 1 < 0)

    log_complete("RANDOM UNIFORM")