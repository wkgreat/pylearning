import numpy as np

'''
N 点数量
x_np (N,2) 所有的点
y_np 每个点的标签
'''

N = 100
x_zeros = np.random.multivariate_normal(mean=np.array((-1,-1)), cov=.1*np.eye(2), size=(N//2,)) #多元正态分布矩阵
y_zeros = np.zeros((N//2,))

x_ones = np.random.multivariate_normal(mean=np.array((1,1)), cov=.1*np.eye(2), size=(N//2))
y_ones = np.ones((N//2,))

x_np = np.vstack([x_zeros, x_ones]) # 矩阵竖向堆叠
y_np = np.concatenate([y_zeros, y_ones]) # 数组连接


def draw():
    import matplotlib.pyplot as plt
    plt.scatter(x_np[:,0],x_np[:,1])
    plt.show()
    pass


if __name__ == '__main__':
    draw()
