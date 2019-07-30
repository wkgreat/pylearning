import numpy as np
from matplotlib import pyplot as plt
import random

def build_func(a=2):
    def the_func(x):
        return a * x

    return the_func

def linear_regression():
    '''
    linear regression
    :return: None
    '''
    target = 10

    def lossFunc(a, x, y):
        return abs(y.T - x.T * a)

    def deltaFunc(a, x, y):
        alpha = 0.0001
        return -1 * 2 * alpha * ((a * x - y) * x.T) / 1000

    x = np.linspace(-100, 100, 1000)
    y = np.array([build_func(target)(xx)+random.random()*1000 for xx in x])
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sp, = ax.plot(x, y)

    a = 0.1
    mx = np.mat(x)
    my = np.mat(y)

    for i in range(20):
        delta = deltaFunc(a, mx, my)
        loss = lossFunc(a, mx, my)
        mean_loss = np.sum(loss) * 1.0 / loss.shape[0]
        a += delta[0, 0]
        print a

        ax.plot(x, np.array([build_func(a)(xx) for xx in x]))

    plt.ioff()
    plt.show()

    print a

if __name__ == '__main__':
    linear_regression()
