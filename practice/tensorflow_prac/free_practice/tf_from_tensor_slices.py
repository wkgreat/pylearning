import tensorflow as tf
import numpy as np


def demo():
    """
    from_tensor_slices:
        x_train (10,5)
        y_train (10,1)
        -> train_ds ((5,1), (1,1)) 相当于每一行进行配对
    """

    x_train = np.random.randint(0, 100, (10, 5))
    y_train = np.random.randint(0, 10, (10, 1))
    print(x_train)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle()

    print(train_ds)

if __name__ == '__main__':
    demo()