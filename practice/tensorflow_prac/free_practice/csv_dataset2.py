"""
dataset 中的每个条目都是一个批次，用一个元组（多个样本，多个标签）表示。
样本中的数据组织形式是以列为主的张量（而不是以行为主的张量），每条数据中包含的元素个数就是批次大小
"""
import tensorflow as tf


def make_csv_dataset(file_path):
    LABEL_COLUMN='y'
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=2,  # 为了示例更容易展示，手动设置较小的值
        label_name=LABEL_COLUMN, # 标签列
        na_value="?",
        ignore_errors=True,
        header=True,  # 第一行是不是列名，
        shuffle=False  # 是否混洗
    )
    return dataset


def main():

    train_data = make_csv_dataset(r'./test.csv')
    train_data_iter = iter(train_data)
    while True:
        train_data, train_label = next(train_data_iter)
        print(train_data)
        print(train_label)


if __name__ == '__main__':
    main()


