import numpy as np 


x = [[1, 2], [1, -3]]
y = [[1, 2], [1, -3]]


x = [[1, 2, 3], [4, 5, 6]]
y = [[1, 2], [3, 4], [5, 6]]

x = np.array(x).astype(np.float64)
y = np.array(y).astype(np.float64)
print (y)
print (y.T)
print (x.shape)
print (y.shape)

print (np.matmul(x, y))