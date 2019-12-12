import tensorflow
tf = tensorflow.compat.v1
tf.disable_eager_execution()

def read_csv():
    filepath = r'./csv_dataset.py'
    file_queue = tf.train.string_input_producer([filepath], num_epochs=1)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.]]  # 为每个字段设置初始值
    cvscolumn = tf.decode_csv(value, defaults)  # 为每一行进行解析

    featurecolumn = [i for i in cvscolumn[1:-1]]
    labelcolumn = cvscolumn[-1]

    with tf.Session() as sess:
        a = sess.run(featurecolumn)
        print(a)

    print("Over.")


if __name__ == '__main__':
    read_csv()
