"""
读取CSV文件
"""
import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()

def read_csv(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0.], [0.], [0.], [0.], [0.]]  # 为每个字段设置初始值
    cvscolumn = tf.decode_csv(value, defaults)  # 为每一行进行解析

    featurecolumn = [i for i in cvscolumn[0:-1]] #每行最后一个元素为标签
    labelcolumn = cvscolumn[-1]

    return featurecolumn, labelcolumn

def create_pipeline(filename, batch_size, num_epochs=None):
    """创建队列数据集函数"""
    # 创建一个输入队列
    file_queue = tf.train.string_input_producer([filename])
    feature, label = read_csv(file_queue)
    min_after_dequeue = 1
    batch_size = 2
    capacity = min_after_dequeue + batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


if __name__ == '__main__':
    with tf.Session() as sess:
        feature, label = create_pipeline(r"./test.csv", 1, 2)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while True:
            if coord.should_stop():
                break
            theFeature, theLabel = sess.run([feature, label])
            print(theFeature)
            print(theLabel)
