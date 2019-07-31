import random

def bootstrapping_posibility():

    d = [0]*10000
    for i in xrange(10000):
        d[random.randint(0, len(d) - 1)] = 1
    print 1 - (sum(d) / 10000.0)

if __name__ == '__main__':
    bootstrapping_posibility()