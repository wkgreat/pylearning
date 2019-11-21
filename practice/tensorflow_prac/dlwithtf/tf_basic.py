import tensorflow as tf
import numpy as np

'''
使用eager execution模式进行测试
'''

#初始化张量
def demo1():
    print("===demo1===")

    a = tf.zeros(2) # 值为0的张量
    print(a.numpy())

    b = tf.ones(2) # 值为1的张量
    print(b.numpy())

    c = tf.fill((2,2), value=5.) # 值为指定值的张量
    print(c.numpy())

    d = tf.constant(3) # 常量
    print(d.numpy())

    # 正态分布的张量
    e = tf.random.normal((2,2), mean=0, stddev=1)
    print(e.numpy())

    # 正太分布的张量 产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    f = tf.random.truncated_normal((2,2), mean=0, stddev=1)
    print(f.numpy())

    # 均匀分布的张量
    g = tf.random.uniform((2,2), minval=-2, maxval=2)
    print(g.numpy())


# 张量叠加与缩放
def demo2():
    print("===demo2===")

    c = tf.ones((2,2))
    d = tf.ones((2,2))

    # 张量加法
    e = c + d
    print(e.numpy())

    # 张量数乘
    f = 2 * e
    print(f.numpy())

    c = tf.fill((2,2),2.)
    d = tf.fill((2,2),7.)

    # 张量元素乘法
    e = c * d
    print(e.numpy())


# 矩阵运算
def demo3():
    print("===demo3===")

    # 单位矩阵
    h = tf.eye(4)
    print(h.numpy())

    # 创建序列 并 创建对角矩阵
    r = tf.range(1,5,1)
    print(r.numpy())
    d = tf.linalg.diag(r)
    print(d.numpy())

    # 矩阵转置 tf.linalg.matrix_transpose
    a = tf.ones((2,3))
    print(a.numpy())
    at = tf.linalg.matrix_transpose(a)
    print(at.numpy())

    # 矩阵乘法 tf.matmul
    a = tf.ones((2,3))
    print(a.numpy())
    b = tf.ones((3,4))
    print(b.numpy())
    c = tf.matmul(a,b)
    print(c.numpy())

def demo4():
    print("===demo4===")

    a = tf.ones((2,2), dtype=tf.int32) # 指定数据类型
    print(a.numpy())
    b = tf.dtypes.cast(a, tf.float32) # 数据类型转换
    print(b.numpy())

    a = tf.ones(8)
    print(a.numpy())
    b = tf.reshape(a, (4,2)) # 改变形状
    print(b.numpy())
    c = tf.reshape(a, (2,2,2)) # 改变形状
    print(c.numpy())

    a = tf.ones(2)
    print(a.get_shape()) # 获取张量形状
    print(a.numpy())
    b = tf.expand_dims(a,0) # 扩展维度
    print(b.get_shape())
    print(b.numpy())
    c = tf.expand_dims(a,1) # 扩展维度
    print(c.get_shape())
    print(c.numpy())
    d = tf.squeeze(b) #挤压维度
    print(d.get_shape())
    print(d.numpy())
    e = tf.squeeze(c) #挤压维度
    print(e.get_shape())
    print(e.numpy())





if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
    demo4()