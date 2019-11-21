"""
    tf.data.Dataset: Represents a potentially large set of elements.
    tf.data.Dataset.from_tensor_slices: Creates a Dataset whose elements are slices of the given tensors
    tf.keras.Model:
    tf.keras.layers:
        Dense:
        Flatten
        Conv2D
    tf.keras.metrics.Mean：
        Computes the (weighted) mean of the given values.
    tf.keras.metrics.SparseCategoricalAccuracy：
        Calculates how often predictions matches integer labels.


"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension #WK 类似于变成多波段图像
x_train = x_train[..., tf.newaxis]  # 增加维度，记住用法
print("x_train.shape: {}". format(x_train.shape))
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')  # 卷积 32 (filters) 输出的通道数; 3 size of kernel
        self.conv2 = Conv2D(1, 5, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    # GradientTape: Operations are recorded if they are executed within this context manager
    # and at least one of their inputs is being "watched".
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)  # SparseCategoricalCrossentropy 交叉熵Loss
    gradients = tape.gradient(loss, model.trainable_variables)  # 交叉熵Loss梯度对于可训练变量的梯度
    """
    gradient:
        compute_gradients() 相当于minimize()的第一步，返回(gradient, variable)对的list。
        apply_gradients() minimize()的第二部分，返回一个执行梯度更新的ops。
    """
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 通过Loss梯度下降，优化可训练变量

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 10

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, loss {}, Accuracy {}, Test Loss: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
