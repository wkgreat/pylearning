"""
tf.image 模块进行图片管理
tf.train 模块进行训练集管理
tf.train.slice_input_producer
    是一个tensor生成器，作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
tf.image.resize_image_with_crop_or_pad
    裁剪或将图像填充到目标宽度和高度. 通过集中修剪图像或用零均匀填充图像,将图像调整为目标宽度和高度.
tf.image.per_image_standardization
    线性缩放image以具有零均值和单位范数. 这个操作计算(x - mean) / adjusted_stddev

tf.train.Coordinator
    Coordinator类用来管理在Session中的多个线程
    可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程
    should_stop: returns True if the threads should stop.
    request_stop: requests that threads should stop.
    join: waits until the specified threads have stopped.

tf.train.start_queue_runners
    使用tf.train.start_queue_runners之后，才会启动填充队列的线程，这时系统就不再“停滞”
"""
import os
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle  # sklearn里面的shuffle功能
# tensorflow 版本兼容
tf_main_version = int(tensorflow.__version__.split(".")[0])
tf = tensorflow.compat.v1 if tf_main_version >= 2 else tensorflow
if tf_main_version >=2: tf.disable_eager_execution()


def load_sample(sample_dir):
    """
    读取文件夹中所有的文件路径
    """
    print('loading sample dataset..')
    lfilenames = []
    labelnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        for filename in filenames:
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)
            labelnames.append(dirpath.split(os.sep)[-1])

    lab = list(sorted(set(labelnames)))
    labdict = dict(zip(lab, list(range(len(lab)))))
    labels = [labdict[i] for i in labelnames]
    return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def get_batches(image, label, resize_w, resize_h, channels, batch_size):
    queue = tf.train.slice_input_producer([image, label])  # 实现一个输入队列
    label = queue[1]  # 从输入队列里读取标签

    image_c = tf.read_file(queue[0])  # 从输入队列里读取image路径
    image = tf.image.decode_bmp(image_c, channels)  # 按照路径读取图片
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)  # 修改图片大小
    image = tf.image.per_image_standardization(image)  # 图片标准化
    image_batch, label_batch = tf.train.batch([image, label],  # 生成批次数据
                                              batch_size=batch_size,
                                              num_threads=64)
    images_batch = tf.cast(image_batch, tf.float32)  # 数据类型转换
    labels_batch = tf.reshape(label_batch, [batch_size])  # 修改标签的形状
    return images_batch, labels_batch


def showresult(subplot, title, thisimg):
    p = plt.subplot(subplot)
    p.axis('off')
    p.imshow(np.reshape(thisimg, (28, 28)))
    p.set_title(title)


def showimg(index, label, img, ntop):
    plt.figure(figsize=(20,10))
    plt.axis('off')
    ntop = min(ntop, 9)
    print(index)
    for i in range(ntop):
        showresult(100+10*ntop+1+i, label[i], img[i])
    plt.show()


def main():
    data_dir = r'./mnist_digits_images'
    (image, label), labelnames = load_sample(data_dir)
    print(image)
    batch_size = 16
    image_batches, label_batches = get_batches(image, label, 28, 28, 1, batch_size)
    print(image_batches)
    print(label_batches)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(10):
                if coord.should_stop():
                    break
                images, label = sess.run([image_batches, label_batches])
                showimg(step, label, images, batch_size)
                print(label)
        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()
        coord.join(threads)  # tf.train.Coordinator.join: waits until the specified threads have stopped.


if __name__ == '__main__':
    main()

