import time 
import sail 
import torch 
import numpy as np


batch_size = 256 
inner_size = 32

lin = sail.modules.Linear(inner_size, 64)
lin2 = sail.modules.Linear(64, 1)
sig = sail.modules.Sigmoid()

inputs = np.random.uniform(0, 1, (batch_size, inner_size)).astype(np.float32)
x = sail.Tensor(inputs)
y = sail.Tensor(inputs.T)

times = []
z = sail.matmul(x, y)
for i in range(100000):
    t = time.time()
    
    y = lin(x)
    y = sig(y)
    y = lin2(y)
    y = sig(y)
    # z = sail.divide(x, x)
    # z = sail.matmul(x, y)
    times.append(time.time() - t)

# print (times)

sail_time = np.mean(times)
## torch

lin = torch.nn.Linear(inner_size, 64)
lin2 = torch.nn.Linear(64, 1)
sig = torch.nn.Sigmoid()

inputs = np.random.uniform(0, 1, (batch_size, inner_size)).astype(np.float32)
x = torch.from_numpy(inputs.astype(np.float32))
y = torch.from_numpy(inputs.T.astype(np.float32))
times = []
z = torch.matmul(x, y)
for i in range(100000):
    t = time.time()
    y = lin(x)
    y = sig(y)
    y = lin2(y)
    y = sig(y)
    # z = torch.divide(x, x)
    # z = torch.matmul(x, y)
    times.append(time.time() - t)
    # y.backward(retain_graph=False)
torch_time = np.mean(times)

# print (times)

print ("SAIL: %s | TORCH: %s" % (sail_time, torch_time))
if sail_time < torch_time:
    print ("SAIL is %s%% faster than TORCH" % ((1 - (sail_time/torch_time)) * 100))