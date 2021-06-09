
# import sail, sys, random, os, gc, time

# import numpy as np

# from ..test_utils import *

# def test_matmul():
#     choices_1 = [(10, 20), (300, 500), (1, 2), (2, 1), (8, 8)]
#     choices_2 = [[(20, 10), (20, 3)], [(500, 150)], [(2, 2), (2, 1)], [(1, 2)], [(8, 1), (8, 4)]]
#     times = []
#     for c, ba in zip(choices_1, choices_2):
#         for b in ba:
#             arr1 = np.random.uniform(0, 1, c)
#             arr2 = np.random.uniform(0, 1, b)

#             x1 = sail.Tensor(arr1, requires_grad=False)
#             x2 = sail.Tensor(arr2, requires_grad=False)
            
#             t = time.time()
#             x3 = sail.matmul(x1, x2)
#             times.append(time.time() - t)
#             arr3 = np.matmul(arr1, arr2)

#             assert_approx_equal(np.sum(arr3), np.sum(x3.numpy()), significant=8)

#     log_time(np.mean(times), "MATMUL")

#     return True

# def test_addmm():
#     choices_1 = [(10, 20), (300, 500), (1, 2), (2, 1), (8, 8)]
#     choices_2 = [[(20, 10), (20, 3)], [(500, 150)], [(2, 2), (2, 1)], [(1, 2)], [(8, 1), (8, 4)]]
#     choices_3 = [[(10, 10), (10, 3)], [(300, 150)], [(1, 2), (1, 1)], [(2, 2)], [(8, 1), (8, 4)]]
#     times = []
#     for a, ba, ca in zip(choices_1, choices_2, choices_3):
#         for b, c in zip(ba, ca):
#             arr1 = np.random.uniform(0, 1, a)
#             arr2 = np.random.uniform(0, 1, b)
#             arr3 = np.random.uniform(0, 1, c)

#             x1 = sail.Tensor(arr1, requires_grad=False)
#             x2 = sail.Tensor(arr2, requires_grad=False)
#             x3 = sail.Tensor(arr3, requires_grad=False)
            
#             t = time.time()
#             x4 = sail.addmm(x1, x2, x3)
#             times.append(time.time() - t)
#             arr4 = np.matmul(arr1, arr2) + arr3

#             assert_approx_equal(np.sum(arr4), np.sum(x4.numpy()), significant=8)

#     log_time(np.mean(times), "ADDMM")

#     return True

# def test_tensordot():
#     choices = [
#         {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': 2, 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (4, 3, 2), 'b_shape': (3, 2, 5), 'axes': ([1, 2], [0, 1]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (4, 2, 3), 'b_shape': (3, 5, 2), 'axes': ([2, 1], [0, 2]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (2, 4, 3), 'b_shape': (5, 3, 2), 'axes': ([2, 0], [1, 2]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (2, 3, 4), 'b_shape': (5, 2, 3), 'axes': ([1, 0], [2, 1]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (3, 2, 4), 'b_shape': (2, 5, 3), 'axes': ([0, 1], [2, 0]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (3, 4, 2), 'b_shape': (2, 3, 5), 'axes': ([0, 2], [1, 0]), 'gc_shape': (4, 5)},  # NOQA
#         {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': 1, 'gc_shape': (3, 4, 5, 6)},  # NOQA
#         {'a_shape': (3, 4, 2), 'b_shape': (2, 5, 6), 'axes': ([2], [0]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
#         {'a_shape': (3, 2, 4), 'b_shape': (5, 2, 6), 'axes': ([1], [1]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
#         {'a_shape': (2, 3, 4), 'b_shape': (5, 6, 2), 'axes': ([0], [2]), 'gc_shape': (3, 4, 5, 6)},  # NOQA
#         {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': 2, 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 2, 6), 'axes': ([2, 3], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (4, 5, 2, 3), 'b_shape': (3, 6, 2), 'axes': ([3, 2], [0, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (4, 2, 5, 3), 'b_shape': (6, 3, 2), 'axes': ([3, 1], [1, 2]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (2, 4, 5, 3), 'b_shape': (6, 2, 3), 'axes': ([3, 0], [2, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (2, 4, 3, 5), 'b_shape': (2, 6, 3), 'axes': ([2, 0], [2, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (2, 3, 4, 5), 'b_shape': (2, 3, 6), 'axes': ([1, 0], [1, 0]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (3, 2, 4, 5), 'b_shape': (3, 2, 6), 'axes': ([0, 1], [0, 1]), 'gc_shape': (4, 5, 6)},  # NOQA
#         {'a_shape': (3, 2, 5, 4), 'b_shape': (3, 6, 2), 'axes': ([0, 1], [0, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
#         {'a_shape': (3, 5, 2, 4), 'b_shape': (6, 3, 2), 'axes': ([0, 2], [1, 2]), 'gc_shape': (5, 4, 6)},  # NOQA
#         {'a_shape': (5, 3, 2, 4), 'b_shape': (6, 2, 3), 'axes': ([1, 2], [2, 1]), 'gc_shape': (5, 4, 6)},  # NOQA
#         {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': 3, 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (5, 4, 3, 2), 'b_shape': (4, 3, 2, 6), 'axes': ([1, 2, 3], [0, 1, 2]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (5, 4, 2, 3), 'b_shape': (4, 3, 6, 2), 'axes': ([1, 3, 2], [0, 1, 3]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (5, 2, 4, 3), 'b_shape': (4, 6, 3, 2), 'axes': ([2, 3, 1], [0, 2, 3]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (2, 5, 4, 3), 'b_shape': (4, 6, 2, 3), 'axes': ([2, 3, 0], [0, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (2, 5, 3, 4), 'b_shape': (6, 4, 2, 3), 'axes': ([3, 2, 0], [1, 3, 2]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (2, 3, 5, 4), 'b_shape': (6, 2, 4, 3), 'axes': ([3, 1, 0], [2, 3, 1]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (3, 2, 5, 4), 'b_shape': (6, 2, 3, 4), 'axes': ([3, 0, 1], [3, 2, 1]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (3, 2, 4, 5), 'b_shape': (2, 6, 3, 4), 'axes': ([2, 0, 1], [3, 2, 0]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (3, 4, 2, 5), 'b_shape': (2, 3, 6, 4), 'axes': ([1, 0, 2], [3, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (4, 3, 2, 5), 'b_shape': (2, 3, 4, 6), 'axes': ([0, 1, 2], [2, 1, 0]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (4, 3, 5, 2), 'b_shape': (3, 2, 4, 6), 'axes': ([0, 1, 3], [2, 0, 1]), 'gc_shape': (5, 6)},  # NOQA
#         {'a_shape': (4, 5, 3, 2), 'b_shape': (3, 4, 2, 6), 'axes': ([0, 2, 3], [1, 0, 2]), 'gc_shape': (5, 6)},  # NOQA
#     ]

#     for c in choices:
#         arr1 = np.random.uniform(0, 1, c["a_shape"])
#         arr2 = np.random.uniform(0, 1, c["b_shape"])

#         x1 = sail.Tensor(arr1, requires_grad=False)
#         x2 = sail.Tensor(arr2, requires_grad=False)

#         arr3 = np.tensordot(arr1, arr2, axes=c["axes"])
#         x3 = sail.tensordot(x1, x2, axes=c["axes"])

#         assert_approx_equal(np.sum(arr3), np.sum(x3.numpy()), significant=5)
#     log_complete("TENSORDOT")
#     return True

    