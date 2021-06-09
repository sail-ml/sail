
# import sail, sys, random, os, gc, time

# import numpy as np
# from ..test_utils import *


# def numel(x):
#     s = x.shape 
#     o = 1
#     for i in s:
#         o *= i 
#     return o

# def dictionary_to_vector(dictionary):
#     return np.concatenate([dictionary[i].ravel() for i in dictionary])

# def vector_to_dictionary(vector, dictionary):
#     nd = {}
#     start = 0
#     for d in dictionary:
#         nd[d] = None
#         data = dictionary[d]
#         end = start + int(numel(data))   
#         nd[d] = vector[start:end].reshape(data.shape)
#         start += int(numel(data))   
#     return nd


# dic = {"a": np.random.uniform(1, 2, (35)), 
#        "b": np.random.uniform(1, 2, (35)),
#        }

# def forward(a, b):
#     c = sail.multiply(a, b)
#     d = sail.sum(c)
#     return d


# def check_gradients_vector(forward_fcn, param_dictionary, eps=1e-3):
#     params = [sail.Tensor(param_dictionary[a], requires_grad=True) for a in param_dictionary]
#     output = forward_fcn(*params)
#     output.backward()
#     grads_dic = {}
#     for i, p in enumerate(param_dictionary):
#         grads_dic[p] = params[i].grad.numpy() 

#     parameters = dictionary_to_vector(param_dictionary)
#     grads = dictionary_to_vector(grads_dic)
#     num_params = len(parameters)

#     j_plus = np.zeros(num_params)
#     j_minus = np.zeros(num_params)
#     grad_approx = np.zeros(num_params)

#     for i in range(len(parameters)):
#         # pass
#         params_plus = np.copy(parameters)
#         params_plus[i] += eps 
#         params_plus_dict = vector_to_dictionary(params_plus, param_dictionary)
#         params_plus_ = [sail.Tensor(params_plus_dict[a]) for a in params_plus_dict]
        
#         z = forward_fcn(*params_plus_)
#         j_plus[i] = z.numpy()

#         params_minus = np.copy(parameters)
#         params_minus[i] -= eps 
#         params_minus_dict = vector_to_dictionary(params_minus, param_dictionary)
#         params_minus_ = [sail.Tensor(params_minus_dict[a]) for a in params_minus_dict]
        
#         z = forward_fcn(*params_minus_)
#         j_minus[i] = z.numpy()

#         grad_approx[i] = (j_plus[i] - j_minus[i])/(2 * eps)
#         # grad_approx[i] = grad_approx[i]#to_significant(grad_approx[i], significant=7)


#     num = np.linalg.norm(grad_approx - grads)
#     denom = np.linalg.norm(grad_approx) + np.linalg.norm(grads)
#     diff = num/denom
#     return diff 

# def test_linear_grad():
#     global lin
#     def forward(a):
#         a.requires_grad = True
#         global lin
#         b = lin(a)
#         c = sail.sum(b)
#         return c

#     choices = [(32, 4)]#, (12, 32), (13, 15)]
#     times = []
#     eps = 1e-2
#     for c in choices:

#         arr1 = np.random.uniform(0, 1, (c)).astype(np.float32)
#         lin = sail.modules.Linear(arr1.shape[1], 2, use_bias=True)

#         dic = {
#             "a": arr1,
#         }

#         diff = check_gradients_vector(forward, dic, eps=eps)
#         assert diff < eps

#     log_complete("LINEAR GRAD")
#     return True
  

  
# def test_sigmoid_grad():
#     global sig
#     def forward(a):
#         global sig
#         b = sig(a)
#         c = sail.sum(b)
#         return c

#     choices = [(32, 64), (12, 32), (13, 15)]
#     times = []
#     eps = 1e-2
#     for c in choices:

#         arr1 = np.random.uniform(0, 1, (c)).astype(np.float32)
#         sig = sail.modules.Sigmoid()

#         dic = {
#             "a": arr1,
#         }

#         diff = check_gradients_vector(forward, dic, eps=eps)
#         assert diff < eps

#     log_complete("SIGMOID GRAD")
#     return True
  
  
