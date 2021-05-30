import time 
import torch 
import sail 
import numpy as np

def set_jit_enabled(enabled: bool):
    """ Enables/disables JIT """
    if torch.__version__ < "1.7":
        torch.jit._enabled = enabled
    else:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()


set_jit_enabled(False)




## sail 

lin = sail.modules.Linear(32, 256)
lin2 = sail.modules.Linear(64, 1)
sig = sail.modules.Sigmoid()

inputs = np.random.uniform(0, 1, (16, 32)).astype(np.float32)
x = sail.Tensor(inputs)
# y = sail.Tensor(inputs.T)

times = []
for i in range(100):
    t = time.time()
    y = lin(x)
    # y = sig(y)
    # y = lin2(y)
    # z = sail.divide(x, x)
    # z = sail.matmul(x, y)
    times.append(time.time() - t)

# print (times)

sail_time = np.mean(times)
## torch

lin = torch.nn.Linear(32, 64)
lin2 = torch.nn.Linear(64, 1)
sig = torch.nn.Sigmoid()

inputs = np.random.uniform(0, 1, (16, 32)).astype(np.float32)
x = torch.from_numpy(inputs.astype(np.float32))
# y = torch.from_numpy(inputs.T.astype(np.float32))
times = []
for i in range(100):
    lin = torch.nn.Linear(32, 256)
    lin2 = torch.nn.Linear(64, 1)
    t = time.time()
    y = lin(x)
    # y = sig(y)
    # y = lin2(y)
    # z = torch.divide(x, x)
    # z = torch.matmul(x, y)
    times.append(time.time() - t)
    # y.backward(retain_graph=False)
torch_time = np.mean(times)
# print (times)

print ("SAIL: %s | TORCH: %s" % (sail_time, torch_time))
if sail_time < torch_time:
    print ("SAIL is %sX faster than TORCH" % (torch_time/sail_time))