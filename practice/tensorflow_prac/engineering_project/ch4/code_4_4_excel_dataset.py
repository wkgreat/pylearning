import tensorflow
# tensorflow 版本兼容
tf_main_version = int(tensorflow.__version__.split(".")[0])
tf = tensorflow.compat.v1 if tf_main_version >= 2 else tensorflow
if tf_main_version >=2: tf.disable_eager_execution()


def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.], [0]]  # 为每个字段设置初始值
    cvscolumn = tf.decode_csv(value, defaults)  # 为每一行进行解析

    featurecolumn = [i for i in cvscolumn[1:-1]]
    labelcolumn = cvscolumn[-1]

    return tf.stack(featurecolumn), labelcolumn


def create_pipeline(filename, batch_size, num_epochs=None):
    """创建队列数据集函数"""
    # 创建一个输入队列
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    feature, label = read_data(file_queue)
    min_after_dequeue = 1000  # 在队列里至少保留1000条数据
    capacity = min_after_dequeue + batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


train_path = r'/Users/wkgreat/Documents/base/data/tensorflow_engineering/ch4/p_4_4_excel_data/iris_test.csv'
test_path = r'/Users/wkgreat/Documents/base/data/tensorflow_engineering/ch4/p_4_4_excel_data/iris_training.csv'
x_train_batch, y_train_batch = create_pipeline(train_path, 32, num_epochs=100)
x_test, y_test = create_pipeline(test_path, 32)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化
    local_init_op = tf.local_variables_initializer()  # 初始化本地变量
    sess.run(init_op)
    sess.run(local_init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while True:
            if coord.should_stop():
                break
            example, label = sess.run([x_train_batch, y_train_batch])  # 注入训练数据
            print(example)
            print(label)
    except tf.errors.OutOfRangeError:
        print("Done Reading")
        example, label = sess.run([x_test, y_test])
        print(example)
        print(label)
    except KeyboardInterrupt:
        print("程序终止")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

