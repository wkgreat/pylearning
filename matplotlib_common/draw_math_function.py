import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


def draw(func, xlim=[-1,1], xnum=100):
    '''
    draw a 1-v function curve defined by python function
    :param func: a normal fuction of python. y = func(x)
    :param xlim: [min x, max x] the range of curve to draw
    :param xnum: the nums of points to draw. the Bigger xnum, the smoother curver.
    :return: None
    '''
    X = np.linspace(xlim[0], xlim[1], xnum)
    Y = np.vectorize(func)(X)
    plt.plot(X,Y)
    plt.show()


def draw_func(sym_func, xlim=[-1,1], xnum=100):
    '''
    draw a 1-v function curve defined by sympy function
    :param sym_func: a sympy function. y = sym_func(x)
    :param xlim: [min x, max x] the range of curve to draw
    :param xnum: the nums of points to draw. the Bigger xnum, the smoother curver.
    :return: None
    '''
    X = np.linspace(xlim[0], xlim[1], xnum)
    F = lambda x : sym_func.evalf(subs={'x':x})
    Y = np.vectorize(F)(X)
    plt.plot(X,Y)
    plt.show()

class Function:
    '''
    include common functions defined by sympy
    '''
    x = sym.Symbol('x')
    sigmoid = 1.0/(1+sym.exp(-1*x))
    relu = sym.Max(x,0)

if __name__ == '__main__':
    #draw(sigmoid, [-10,10], 100)
    sigmoid = Function.sigmoid
    d_sig = sym.diff(sigmoid)
    relu = Function.relu
    draw_func(sym.diff(relu), [-10,10], 100)