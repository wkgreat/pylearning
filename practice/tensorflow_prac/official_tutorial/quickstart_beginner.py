"""
    Sequential: Linear stack of layers.
    layers.Flatten: Flattens the input. Does not affect the batch size.
    layers.Dense: Just your regular densely-connected NN layer.
    layers.Dropout: Applies Dropout to the input.


"""
import tensorflow as tf
import matplotlib.pyplot as plt

minist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = minist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

print("x_train.shape: {}, y_train.shape: {}". format(x_train.shape, y_train.shape))

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 输入的变量是一个28*28的图像（表示一个手写数字）
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)