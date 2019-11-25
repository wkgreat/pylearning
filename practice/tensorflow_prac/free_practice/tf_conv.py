"""
练习 使用 tf1 原生框架
TODO 测试失败 后序改进
"""
import numpy as np
from ..engineering_project.ch4 import code_4_3_image_dataset as dataset
import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()

class Model(object):
    """
    images -> conv1 -> pool1 -> relu1 -> conv2 -> pool2 -> relu2 -> flatten -> logits
    """
    def __init__(self, batch_size, height, width, channel):

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel

        self.images = tf.placeholder(tf.float32, shape=(batch_size, height, width, channel))
        self.labels = tf.placeholder(tf.int64, shape=(batch_size,))

        # 这里使用sigmoid效果尚可，如果换成relu就不行了，为什么?
        self.conv1 = tf.nn.sigmoid(self.conv(self.images))
        self.flat1 = self.flattern(self.conv1)
        self.dens1 = tf.nn.sigmoid(self.dense(self.flat1, 512))
        self.dens2 = tf.nn.softmax(self.dense(self.dens1, 10))
        self.logtis = self.dens2

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logtis))

        self.prediction = tf.argmax(self.logtis, axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels), tf.float32))

        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def conv(self, tensor):
        ksize = tf.Variable( tf.random_normal((3, 3, 1, 32), mean=0, stddev=0.1, dtype=tf.float32) )
        strides = [1, 1, 1, 1]
        return tf.nn.conv2d(tensor, ksize, strides, padding='SAME')

    def flattern(self, tensor):
        shape = tensor.get_shape().as_list()
        return tf.reshape(tensor, (shape[0], shape[1]*shape[2]*shape[3]))

    def dense(self, tensor, outdim=32):
        shape = tensor.get_shape().as_list()
        w = tf.Variable(tf.random_normal((shape[1], outdim), mean=0.0, stddev=0.1, dtype=tf.float32))
        b = tf.Variable(tf.random_normal((outdim, ), mean=0.0, stddev=0.1, dtype=tf.float32))
        return tf.matmul(tensor, w) + b

    def predict(self, sess, images):
        return sess.run(self.prediction, feed_dict={self.images: images})

    def train_epoch(self, sess, images, labels):
        # print("RELU1:, ", self.relu1.get_shape())
        # print("RELU2:, ", self.relu2.get_shape())
        # print("RELU:, ", sess.run(self.relu2.get_shape(), feed_dict={self.images: images, self.labels: labels}))
        return sess.run([self.optimizer, self.loss, self.prediction, self.accuracy], feed_dict={self.images: images, self.labels: labels})


def main():
    batch_size = 48
    height = 28
    width = 28
    channel = 1
    data_dir = r'/Users/wkgreat/Documents/base/data/tensorflow_engineering/ch4/p_4_3_image_data/mnist_digits_images'

    model = Model(batch_size, height, width, channel)

    (image, label), labelnames = dataset.load_sample(data_dir)
    image_batches, label_batches = dataset.get_batches(image, label, height, width, channel, batch_size)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(20):
                if coord.should_stop():
                    break
                images, label = sess.run([image_batches, label_batches])
                print("images shape: {}, label shape: {}".format(images.shape, label.shape))
                for i in range(50):
                    _, loss, p_label, accuracy = model.train_epoch(sess, images, label)
                    print("Loss: {}\tAccu: {}".format(loss, accuracy))
                    # print("Label : ", label)
                    # print("PLabel: ", p_label)
            test_images, test_label = sess.run([image_batches, label_batches])
            pred_label = model.predict(sess, test_images)
            print(pred_label)

        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()
        coord.join(threads)  # tf.train.Coordinator.join: waits until the specified threads have stopped.


