import tensorflow as tf
import numpy as np
from dlwithtf import tf_classification_dataset as dataset

tf.compat.v1.disable_v2_behavior()  # 关闭eager模式

'''
logistic classification
Here the cross entropy is Loss Function
原始  X -> Y (0,1)
模型  ~Y = Round(Sigmoid(XW+B))
损失  E = Entropy(~Y,Y)
目标  Arg(W,B) st min E
[1] x(N,2) y(N,) W(2,1) b(1,)
    xW -> (N,2)(2,1) -> (N,1) squeeze (N,)
[2] 对于函数 sigmoid_cross_entropy_with_logits(logits, labels)
    s = sigmoid(t) 在这里logtis相当于t. 所以只需要将t传给该函数就行了，不需要手动构建sigmoid

'''

def main():

    N = dataset.N
    x_np = dataset.x_np
    y_np = dataset.y_np

    with tf.name_scope("placeholders"):
        x = tf.compat.v1.placeholder(tf.float32, (N,2)) # [1]
        y = tf.compat.v1.placeholder(tf.float32, (N,))
    with tf.name_scope("weights"):
        W = tf.Variable(tf.random.normal((2,1)))
        b = tf.Variable(tf.random.normal((1,)))
    with tf.name_scope("prediction"):
        y_logit = tf.squeeze(tf.matmul(x,W)+b) #squeeze 从张量形状中移除大小为1的维度
        y_one_prob = tf.sigmoid(y_logit)
        y_pred = tf.round(y_one_prob) # 对sigmoid四舍五入，相当于分成两类了
    with tf.name_scope("loss"):
        entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y) # [2]
        l = tf.reduce_sum(entropy)
    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(.01).minimize(l)

        tf.compat.v1.summary.scalar("loss", l)
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter("/tmp/logistic-train", tf.compat.v1.get_default_graph())

    n_steps = 1000
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_steps):
            feed_dict = {x: x_np, y: y_np}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
            print("Step %d, loss: %f" % (i, loss))
            train_writer.add_summary(summary, i)


        #test
        pw = sess.run(W)
        pb = sess.run(b)
        print("w: ", pw.flatten())
        print("b: ", pb)
        p_y_logit = np.matmul(x_np,pw) + pb
        p_y_sig = np.vectorize(lambda x: 1.0/x)((1 + np.exp(p_y_logit)))
        p_y_pred = np.round(p_y_sig).flatten()

        print("y_np: \n", y_np)
        print("p_y_pred: \n", p_y_pred)






def test():
    with tf.compat.v1.Session() as sess:
        x_np = tf.dtypes.cast(tf.convert_to_tensor(dataset.x_np), tf.float32)
        y_np = tf.dtypes.cast(tf.convert_to_tensor(dataset.y_np), tf.float32)
        W = tf.random.normal((2, 1))
        b = tf.random.normal((1,))
        y_logit = tf.squeeze(tf.matmul(x_np, W) + b)
        y_one_prob = tf.sigmoid(y_logit)
        y_pred = tf.round(y_one_prob)  # 对sigmoid四舍五入，相当于分成两类了
        print(y_pred.eval(session=sess))
        entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_np)
        l = tf.reduce_sum(entropy)
        print(l.eval(session=sess))

if __name__ == '__main__':
    main()
