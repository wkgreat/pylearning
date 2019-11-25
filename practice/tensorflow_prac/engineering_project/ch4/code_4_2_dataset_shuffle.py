import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle  # sklearn里面的shuffle功能
# tensorflow 版本兼容
tf_main_version = int(tensorflow.__version__.split(".")[0])
tf = tensorflow.compat.v1 if tf_main_version >= 2 else tensorflow
if tf_main_version >=2: tf.disable_eager_execution()


def GenerateData(training_epochs, batchsize=100):
    for i in range(training_epochs):
        train_X = np.linspace(-1, 1, batchsize)
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
        yield shuffle(train_X, train_Y), i


Xinput = tf.placeholder("float", (None))
Yinput = tf.placeholder("float", (None))

train_epochs = 20

with tf.Session() as sess:
    for (x, y), ii in GenerateData(train_epochs):
        xv, yv = sess.run([Xinput, Yinput], feed_dict={Xinput: x, Yinput: y})
        print(ii, "| x.shape:", np.shape(xv), "| x[:3]:", xv[:3])
        print(ii, "| y.shape:", np.shape(yv), "| y[:3]:", yv[:3])


train_data = list(GenerateData(1))[0]
plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
plt.legend()
plt.show()