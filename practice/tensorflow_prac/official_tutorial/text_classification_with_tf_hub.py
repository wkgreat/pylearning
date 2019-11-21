import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager Mode: ", tf.executing_eagerly())
print("Hub Version: ", hub.__version__)
# 判断是否可以用GPU
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
)

train_example_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_example_batch)
print(train_labels_batch)

# Text embedding based on Swivel co-occurrence matrix factorization[1] with pre-built OOV.
# Maps from text to 20-dimensional embedding vectors.
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"

# 句子转换为嵌入向量的Layer
hub_layer = hub.KerasLayer(
    embedding,
    input_shape=[],  # Expects a tensor of shape [batch_size] as input.
    dtype=tf.string,
    trainable=True
)
print(hub_layer(train_example_batch))  # 测试一下

model = tf.keras.Sequential()
model.add(hub_layer)  # 将句子转换为嵌入向量层
model.add(tf.keras.layers.Dense(16, activation='relu'))     # 该定长输出向量通过一个有 16 个隐层单元的全连接层（Dense）进行管道传输
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))   # WK 输出维度为1，激活sigmoid 二分类问题 直接使用逻辑回归分类

print(model.summary())

model.compile(
    optimizer="Adam",               # see `tf.keras.optimizers`
    loss="binary_crossentropy",     # see `tf.losses`
    metrics=['accuracy']            # List of metrics to be evaluated by the model during training and testing
)

history = model.fit(
    train_data.shuffle(10000).batch(512),           # Input data.
    epochs=20,                                      # Number of epochs to train the model
    validation_data=validation_data.batch(512),     # the loss and any model metrics at the end of each epoch. The model
                                                    # will not be trained on this data.
    verbose=1                                       # 0 = silent, 1 = progress bar, 2 = one line per epoch.
)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
