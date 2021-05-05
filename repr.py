import numpy as np 

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
        print (self.strides)
        print (self.shape)
    def next(self):
        for i in range(self.ndim-1, -1, -1):
            if (self.coordinates[i] < (self.shape[i] - 1)):
                self.coordinates[i] += 1
                self.d_ptr += self.strides[i]
                break
            else:
                self.coordinates[i] = 0
                # print ("%s - %s" % (self.d_ptr,self.back_strides[i]))
                self.d_ptr -= self.back_strides[i]

    def get(self):
        print (self.coordinates, self.d_ptr)
        return self.data[self.d_ptr]

    def get_next(self):
        self.next()
        return self.get()


s0 = (3, 3, 2)
s1 = (2)
A=np.arange(np.prod(s0)).reshape(s0)
B=np.arange(np.prod(s1)).reshape(s1)
# A,B=np.lib.stride_tricks.broadcast_arrays(A, B)
# print (B.strides)
# exit()
it = np.nditer((A, B))
z = 0
for _, b in it:
    print (b)
    if z > 12:
        break 
    z += 1
print (" ")


shape = A.shape
strides = (6, 2, 1)
x = np.arange(np.prod(shape)).reshape(shape).flatten()
print (x.reshape(shape))
ar = ArrayIterator(x.tolist(), shape, (3, 3, 2), strides)
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
prev_d = ar.d_ptr
for i in range(len(x)):
    print (ar.get())
    ar.get_next()

    if ar.d_ptr != (prev_d+1):
        print ("move")
    prev_d = ar.d_ptr