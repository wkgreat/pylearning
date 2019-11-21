import deepchem as dc
import tensorflow as tf
from sklearn.metrics import accuracy_score
tfcv = tf.compat.v1
tfcv.disable_eager_execution()


def main():

    # load the tox21 dataset
    # X is factor vector, y is label, w is weight
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # delect additional data
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    d = 1024
    n_hidden = 50
    learning_rate = .001
    n_epochs = 10
    batch_size = 100
    dropout_prob = 1.0

    with tf.name_scope("palceholders"):  # 占位符
        # None 代表指定维度上，可以接受任意长度的张量
        x = tfcv.placeholder(tf.float32, (None, d))
        y = tfcv.placeholder(tf.float32, (None,))
        keep_prob = tfcv.placeholder(tf.float32)
    with tf.name_scope("hidden-layer"):  # 隐含层
        W = tf.Variable(tf.random.normal((d, n_hidden)))
        b = tf.Variable(tf.random.normal((n_hidden,)))
        x_hidden = tf.nn.relu(tf.matmul(x,W)+b)  # RELU激活函数
        x_hidden = tf.nn.dropout(x_hidden, keep_prob)  # dropout 参数keep_prob表示节点保留的概率
    with tf.name_scope("output"):
        W = tf.Variable(tf.random.normal((n_hidden,1)))
        b = tf.Variable(tf.random.normal((1,)))
        y_logit = tf.matmul(x_hidden, W) + b
        y_one_prob = tf.sigmoid(y_logit)
        y_pred = tf.round(y_one_prob)
    with tf.name_scope("loss"):
        y_expand = tf.expand_dims(y,1)
        entropy = tfcv.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
        l = tf.reduce_sum(entropy)
    with tf.name_scope("optim"):
        train_op = tfcv.train.AdamOptimizer(learning_rate).minimize(l)
    with tf.name_scope("summaries"):
        tfcv.summary.scalar("loss", l)
        merged = tfcv.summary.merge_all()

    train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21', tf.get_default_graph())

    N = train_X.shape[0]
    with tfcv.Session() as sess:
        sess.run(tfcv.global_variables_initializer())
        step = 0
        for epoch in range(n_epochs):
            pos = 0
            while pos < N:
                batch_X = train_X[pos:pos + batch_size]
                batch_y = train_y[pos:pos + batch_size]
                feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
                _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                train_writer.add_summary(summary, step)

                step += 1
                pos += batch_size

        # Make Predictions (set keep_prob to 1.0 for predictions) 进行预测
        train_y_pred = sess.run(y_pred, feed_dict={x: train_X, keep_prob: 1.0})
        valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
        test_y_pred = sess.run(y_pred, feed_dict={x: test_X, keep_prob: 1.0})

    # 评估模型精度
    train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
    valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    print("Valid Weighted Classification Accuracy: %f" % valid_weighted_score)
    test_weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
    print("Test Weighted Classification Accuracy: %f" % test_weighted_score)


if __name__ == '__main__':
    main()



