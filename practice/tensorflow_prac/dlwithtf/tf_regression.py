import tensorflow as tf
from dlwithtf import tf_regression_dataset as dataset

tf.compat.v1.disable_v2_behavior()  # 关闭eager模式

'''
linear regression
'''

def main():

    N = dataset.N

    with tf.name_scope("placeholders"):

        x = tf.compat.v1.placeholder(tf.float32, (N,1))
        y = tf.compat.v1.placeholder(tf.float32, (N,1))

    with tf.name_scope("weights"):

        W = tf.Variable(tf.random.normal((1,1)))
        b = tf.Variable(tf.random.normal((1,)))

    with tf.name_scope("prediction"):

        y_pred = tf.matmul(x, W) + b

    with tf.name_scope("loss"):

        l = tf.reduce_sum((y - y_pred)**2)

    with tf.name_scope("optim"):

        train_op = tf.compat.v1.train.AdamOptimizer(.005).minimize(l)

    with tf.name_scope("summaries"):

        tf.compat.v1.summary.scalar("loss", l)
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter("/tmp/lr-train", tf.compat.v1.get_default_graph())

    n_steps = 10000
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_steps):
            feed_dict = {x: dataset.x_np, y: dataset.y_np}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("Step %d, loss: %f" % (i, loss))
            train_writer.add_summary(summary, i)

        w_pred = sess.run(W)
        b_pred = sess.run(b)

    # visualization
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset.x_np, dataset.y_np)
    ax.plot(dataset.x_np, dataset.x_np * w_pred[0][0] + b_pred[0])
    plt.show()


if __name__ == '__main__':
    main()