import tensorflow as tf

def main():
    a = tf.to_int32(tf.random_uniform([1, 10, 10, 3], 0, 255))
    with tf.Session() as sess:
        print(sess.run(a))

#conv2d
def demo1():
    input = tf.round(tf.random_uniform([1, 10, 10, 3], 0, 255))
    filter = tf.ones((3, 3, 3, 1))
    strides = [1, 1, 1, 1]
    padding = "SAME"
    conv = tf.nn.conv2d(input, filter, strides, padding)
    with tf.Session() as sess:
        print(sess.run(conv))

#data_format
# The Conv2D op currently only supports the NHWC tensor format on the CPU
def demo2():
    with tf.Session() as sess:

        input = tf.round(tf.random_uniform([1, 5, 5, 1], 0, 10))  #输入张量
        input_r = tf.transpose(input, [0, 3, 1, 2])  #显示用

        filter = tf.ones((3, 3, 1, 3))  # 卷积核
        strides = [1, 1, 1, 1]  #步长
        padding = "SAME"  # 边界填充
        data_format = "NHWC"  # 维度顺序 NHWC (num, height, width, chennel)
        conv = tf.nn.conv2d(input, filter, strides, padding, data_format=data_format)  # 二维卷积
        conv_r = tf.transpose(conv, [0, 3, 1, 2])

        input_r, conv_r = sess.run([input_r, conv_r])

        print(input_r)
        print(conv_r)

#pool
def demo3():
    """
    shape:
        value（1,5,5,1）
        ksize (1,2,2,1)
        strides (1,2,2,1)
        pool (1,3,3,1)
    """
    with tf.Session() as sess:
        value = tf.round(tf.random_uniform([1,5,5,1], 0, 255))
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = "SAME"

        pool = tf.nn.max_pool(value, ksize, strides, padding)
        pool_shape = tf.Variable(pool.get_shape())

        value_r = tf.transpose(value, [0, 3, 1, 2])  #显示用
        pool_r = tf.transpose(pool, [0, 3, 1, 2])
        sess.run(tf.global_variables_initializer())
        value_r, pool_r, pool_shape = sess.run([value_r, pool_r, pool_shape])
        print(pool_shape)
        print(value_r)
        print(pool_r)


if __name__ == '__main__':
    demo3()
