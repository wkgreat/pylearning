import numpy as np

a = np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,0,1,1]])
b = np.array([12,14,31,34]).reshape(4,1)
x = np.matmul(np.linalg.inv(a),b).reshape(1,4)
print(x[0,1] + x[0,3])