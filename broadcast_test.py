
import numpy as np


# def broadcast_add(a, b, a_shape, b_shape, a_strides, b_strides, a_numel):

#     i_0 = 0
#     i_1 = 0

#     d_0 = len(a_shape) - 1
#     d_1 = len(b_shape) - 1

#     out = []

#     ## assumes we can broadcast

#     for i_0 in range(a_numel):
#         out.append(a[i_0] + b[i_1])

#         i_1 += 

#         # i_1 += 1
#         # if i_1 >= len(b):
#         #     i_1 = 0

#     return out

# a_shape = (2, 2, 2)
# b_shape = (2, 1, 2)

# a_strides = (2, 2, 1)
# b_strides = (2, 0, 1)

# a_n = np.arange(np.prod(a_shape)).reshape(a_shape)
# b_n = np.arange(np.prod(b_shape)).reshape(b_shape)

# sol = a_n + b_n 
# sol = sol.flatten()

# a = a_n.flatten().tolist()
# b = b_n.flatten().tolist()

# a_numel = len(a)

# x = broadcast_add(a, b, a_shape, b_shape, a_numel)
# # print (sol)
# # assert(sum(x) == sum(sol))
# for a, b in zip(x, sol):
#     print (a, b)


class ArrayIterator():

    def __init__(self, data, shape, treat_shape, strides):
        self.data = data 
        self.shape = shape 
        self.ts = treat_shape
        self.ndim = len(shape)
        self.strides = strides 
        self.coordinates = [0] * (self.ndim)
        self.d_ptr = 0
        self.back_strides = [self.strides[i] * (self.shape[i] - 1) for i in range(self.ndim)]
        print (self.shape)
        print (self.strides)
        print (self.back_strides)
        self.shape = (4, 4, 4)
        self.strides = (4, 0, 1)
        self.back_strides = (12, 0, 3)
    def next(self):
        for i in range(self.ndim-1, -1, -1):
            print (self.coordinates[i], 3)
            if (self.coordinates[i] < (self.shape[i] - 1)):
                self.coordinates[i] += 1
                self.d_ptr += self.strides[i]
                break
            else:
                self.coordinates[i] = 0
                # print ("%s - %s" % (self.d_ptr,self.back_strides[i]))
                self.d_ptr -= self.back_strides[i]

    def get(self):
        print ("DP", self.d_ptr)
        print (self.coordinates)
        return self.data[self.d_ptr]

    def get_next(self):
        self.next()
        return self.get()

            
A=np.arange(np.prod((4, 4, 4))).reshape(4,4,4)
B=np.arange(np.prod((4, 1, 4))).reshape(4, 1, 4)
# A,B=np.lib.stride_tricks.broadcast_arrays(A, B)

it = np.nditer((A, B))
z = 0
for _, b in it:
    print (b)
    if z > 12:
        break 
    z += 1
print (" ")


shape = A.shape
strides = (4, 0, 1)
x = np.arange(np.prod(shape)).reshape(shape).flatten().tolist()

ar = ArrayIterator(x, shape, (2, 2, 2), strides)
# print (ar.get()) # 0
# print (ar.get_next()) # 1
# print (ar.get_next()) # 2
# print (ar.get_next()) # 3
# print (ar.get_next()) # 4
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
# print (ar.get_next()) # 5
for i in range(len(x)):
    ar.get_next()