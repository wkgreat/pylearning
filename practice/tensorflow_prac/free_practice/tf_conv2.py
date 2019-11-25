"""
练习 使用tf2.0 keras API
"""
import numpy as np
from ..engineering_project.ch4 import code_4_3_image_dataset as dataset
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, 1, 'same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def main():
    batch_size = 100
    height = 28
    width = 28
    channel = 1
    data_dir = r'/Users/wkgreat/Documents/base/data/tensorflow_engineering/ch4/p_4_3_image_data/mnist_digits_images'

    (image, label), labelnames = dataset.load_sample(data_dir)
    image_batches, label_batches = dataset.get_batches(image, label, height, width, channel, batch_size)

    with tf.compat.v1.Session() as sess:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(100):
                if coord.should_stop():
                    break
                images, label = sess.run([image_batches, label_batches])
                model.fit(images, label)

        except tf.errors.OutOfRangeError:
            print("Done!!!")
        finally:
            coord.request_stop()
        coord.join(threads)  # tf.train.Coordinator.join: waits until the specified threads have stopped.

if __name__ == '__main__':
    main()