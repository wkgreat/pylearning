import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

"""
梯度功能练习
"""


def prac_gredient():
    """
    梯度下降
    :return:
    """
    x = tf.Variable(100.0, tf.float32)
    y = x * x
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(y)

    ix = []
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100):
            _, vx = sess.run([optimizer, x])
            print("step: {}, x: {}".format(i, vx))
            ix.append(float(vx))
    iy = list(map(lambda x: x*x, ix))
    plt.plot(np.array(ix), np.array(iy))
    plt.show()



if __name__ == '__main__':
    prac_gredient()

