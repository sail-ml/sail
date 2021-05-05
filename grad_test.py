import sail 
import numpy as np
import cupy as cp
import torch
import resource

# x = torch.randn(2, 4, requires_grad=True)
# y = torch.randn(4, 6, requires_grad=True)
# z = x.mm(y)
# print (x)
# print (y)
# print (torch.autograd.grad(torch.sum(z), x)[0])
# exit(0)

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


dic = {"a": np.random.uniform(0, 1, (10)), 
       "b": np.random.uniform(0, 1, (10)),
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

    print (grad_approx.shape)
    print (grads.shape)

    num = np.linalg.norm(grad_approx - grads)
    denom = np.linalg.norm(grad_approx) + np.linalg.norm(grads)
    diff = num/denom
    return diff 

diff = check_gradients_vector(forward, dic)
if diff < 1e-7:
    print ("gradient is correct, %s" % diff)
else:
    print ("gradient is incorrect, %s" % diff)

# dic2 = vector_to_dictionary(vector, dic)



# print (a.grad)
# print (b.grad)
