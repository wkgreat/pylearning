import tensorflow as tf

tf.compat.v1.disable_v2_behavior()  # 关闭eager模式

#占位符
def demo1():
    p = tf.compat.v1.placeholder(tf.float32, shape=(2,2))
    print(p)

#Feed字典
def demo2():
    a = tf.compat.v1.placeholder(tf.float32, shape=(1,))
    b = tf.compat.v1.placeholder(tf.float32, shape=(1,))
    c = a + b
    with tf.compat.v1.Session() as sess:
        c_eval = sess.run(c, {a: [1.], b: [2.]}) #输入Feed字典 映射到对应的placeholder
        print(c_eval)

#作用域
def demo3():
    N = 5
    with tf.name_scope("placeholders"): #作用域
        x = tf.compat.v1.placeholder(tf.float32, (N,1))
        y = tf.compat.v1.placeholder(tf.float32, (N,))
    print(x)


#优化器
def demo4():
    W = tf.Variable((3,), dtype=tf.float32)
    l = tf.reduce_sum(W) # 求和 示意性的loss
    learning_rate = .001
    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(l)
        print(train_op)


# 梯度下降
def demo5():
    W = tf.Variable((3,))
    l = tf.reduce_sum(W) # 求和
    gradW = tf.gradients(l,W)
    print(gradW)

    # 日志
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", l)
        merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter("/tmp/lr-train", tf.compat.v1.get_default_graph())



if __name__ == '__main__':
    demo4()
