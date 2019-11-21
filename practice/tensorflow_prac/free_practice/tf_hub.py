import tensorflow as tf
import tensorflow_hub as hub

def demo1():
    """
    调用句子转嵌入向量的模块
    :return:
    """
    embed = hub.load("https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1")
    embeddings = embed(["cat is on the mat", "dog is in the fog"])
    print(embeddings)


if __name__ == '__main__':
    demo1()