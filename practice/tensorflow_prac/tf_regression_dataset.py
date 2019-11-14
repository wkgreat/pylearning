import numpy as np
N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N,1)
noise = np.random.normal(scale=noise_scale, size=(N,1))
y_np = np.reshape(w_true * x_np + b_true + noise, (N,1))


def draw():
    import matplotlib.pyplot as plt
    plt.scatter(x_np,y_np)
    plt.show()
    pass

if __name__ == '__main__':
    draw()
