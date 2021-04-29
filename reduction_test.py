import numpy as np 


# x = np.random.uniform(0, 1, (3, 2, 3, 5))

# print (np.sum(x, 1))
# x = x.reshape(3, 3, 5, 2)
# print (np.sum(x, -1))


# def pairwise_sum(a):

#     def pairwise_inner(b):
#         x = len(b)
#         if x == 1:
#             return b[0]

#         x = pairwise_inner(b[:x//2])
#         y = pairwise_inner(b[x//2:])
#         return x + y 

#     return pairwise_inner(a) 

# print (pairwise_sum([1, 2, 3, 4]))
        


# def sail_sum(a, shape, strides, axis):
#     sum_stride = int(np.prod(shape[1+axis:]))
#     element_cout = shape[axis]
#     print ("ELEMENT COUNT", element_cout)
#     print ("SUM STRIDE", sum_stride)
#     out = []
#     print (len(a)//element_cout)
#     its = len(a)//(sum_stride * element_cout)
#     print ("ITS", its)
#     z = 0
    # for i in range(np.prod(shape[:axis] + shape[axis+1:])):
    # while z != len(a):
    #     for j in range(element_cout):
    #         s = 0
    #         # for k in range(sum_stride):
    #             # s += a[(z+k) + sum_stride * j]
    #         print ((z) + sum_stride * j)
    #         # print (s)
    #     z += 1

#     for i in range(0, len(a), sum_stride * element_cout):
#         s = 0
#         for j in range(sum_stride):
#             print ((i + j), ((i + j) + sum_stride))
#             # s = a[i + j] + a[(i + j) + sum_stride]
#             # out.append(s)
#         # if z%element_cout == 0:
#             # z += element_cout
#         # z = i
#         # for j in range(sum_stride):
#         #     s = 0
#         #     for k in range(element_cout):
#         #         s += a[(z + j) + (sum_stride * k)]
#         #     out.append(s)

#     print (out)
#             # z += sum_stride


import numpy as np

a = np.arange(15  * 4 * 14).reshape(15, 4,14)
print (a)
shape = a.shape

b = a.flatten()
print (b)

axis = 2
ms = shape[axis]

r_j = int(np.prod(shape[axis+1:]))
print ("IC", r_j)
o = []#[0] * len(b)
idx = 0
inner_count = 0
insert_idx = 0
print (np.sum(a, axis))
while idx < len(b):
    # print (b[idx], idx, "add to", b[idx+r_j])
    # print ([b[idx + (r_j * i)] for i in range(ms)])
    o.append(sum([b[idx + (r_j * i)] for i in range(ms)]))
    inner_count += 1
    idx += 1
    insert_idx += 1
    if inner_count == (r_j):
        idx += (r_j * (ms - 1)) 
        inner_count = 0
        # print (" ")

print (np.array(o).reshape(np.sum(a, axis).shape))