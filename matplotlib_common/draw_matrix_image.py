import matplotlib.pyplot as plt


def draw_matrix_image(m, label=None):

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(m)

    if(label):
        # plt.text(2, 2, label, color='red', fontsize=20)  #put a text label
        plt.xlabel(label)
    plt.show()


def _demo1():
    import numpy as np
    m = np.random.randint(0, 256, size=(28,28), dtype=np.int)
    draw_matrix_image(m, "WK")


if __name__ == '__main__':
    _demo1()