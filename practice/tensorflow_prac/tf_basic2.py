from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

#广播
def demo1():
    print("===demo1===")
    a = tf.ones((2,2))
    print(a.numpy())
    b = tf.range(0,2,1, dtype=tf.float32) #tf不执行隐式的类型转换，所以要声明dtype
    print(b.numpy())
    c = a + b #向量b被加到矩阵a的每一行当中
    print(c.numpy())

#图
def demo2():
    print("===demo2===")
    print(tf.compat.v1.get_default_graph()) # 获取默认图

#会话
def demo3():
    print("===demo3===")
    tf.compat.v1.disable_v2_behavior() #关闭eager模式
    a = tf.ones((2,2))
    b = tf.matmul(a,a)
    sess = tf.compat.v1.Session() #获取Session
    print(b.eval(session=sess)) #等价于 print(sess.run(b))


#变量 变量必须显式初始化
def demo4():
    print("===demo4===")
    tf.compat.v1.disable_v2_behavior()  # 关闭eager模式
    a = tf.Variable(tf.ones((2,2))) # 创建变量
    sess = tf.compat.v1.Session() # 创建Session
    sess.run(tf.compat.v1.global_variables_initializer()) # 变量必须显式初始化
    r = a.eval(session=sess)
    print(r)

    # assgin 对变量进行赋值。
    # 注意：变量在初始化后形状就固定了，所以赋值时必须形状相等
    r = sess.run(a.assign(tf.zeros((2,2))))
    print(r)


    pass
if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()