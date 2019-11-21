import sympy as sym
import sympy.vector as vec
from  functools import reduce
import math

class Functions(object):
    """
    include common functions defined by sympy
    """

    x = sym.Symbol('x')
    v1 = vec.Vector('v1')
    v2 = vec.Vector('v2')

    sigmoid = 1.0/(1+sym.exp(-1*x))

    relu = sym.Max(x, 0)


def sigmoid(x):
    """ x numble """
    return 1.0 / (1 + sym.exp(-1 * x))


def softmax(X):
    """ X iterable list """
    s = sum(map(lambda x: math.exp(x), X))
    return [math.exp(x)/s for x in X]


def cross_entropy(X1, X2):
    """ X1, X2 iterable list """
    return - sum([(X1[i]*math.log(X2[i])) for i in range(min(len(X1), len(X2)))])


def softmax_cross_entropy(X, Y):
    return cross_entropy(Y,softmax(X))


if __name__ == '__main__':
    y = [1,0,0,0,0]
    output = [1,0,0,0,0]
    print(softmax_cross_entropy(output, y))



