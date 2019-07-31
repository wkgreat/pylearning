"""
KE WANG | wkgreat@outlook.com
20190731
KMeans Algorithm
"""
import random
import numpy as np
from copy import deepcopy


class LabelPoint:
    """
    the labeled point or (feature vector)
    point is multi-dimensional vector
    label is a unique identity of which cluster this point belong
    """

    def __init__(self, point, label):
        self.point = np.array(point)
        self.label = label

    def __str__(self):
        return "[Point: %r, Label: %s]" % (self.point, self.label)


class Kmeans():

    def __init__(self, k, max_iter=10000):
        """
        :param k: the k value, how many clusters
        :param max_iter: max num of iteration of kmeans algorithm
        """
        self.max_iter = max_iter
        self.k = k
        self.points = []
        self.old_centers = {}
        self.centers = {}

    def set_sample(self, points):
        """
        :param points: the sample points for traning. [type] python list or numpy array
        :return: None
        """
        self.points = [LabelPoint(p, "") for p in points]
        self.init_centers()

    def init_centers(self):
        self.centers = {i: LabelPoint(p.point, i) for i, p in enumerate(random.sample(self.points, self.k))}

    def train(self):
        """
        train the kmeans model
        :return:
        """
        i = 0
        while i <= self.max_iter:
            self.cluster()
            self.old_centers = deepcopy(self.centers)
            self.center()
            m = self.total_center_dist()
            if m < 0.001:
                break

    def label(self, p):
        """
        label a point(vector) by the cluster which has min distance to it.
        :param p: the point
        :return: the cluster label
        """
        min_dist = 0xFFFFFF
        the_label = ""

        for clabel, center in self.centers.items():
            d = self.dist(p, center)
            if d < min_dist:
                min_dist = d
                the_label = clabel
        return the_label

    def cluster(self):
        """
        recluster label of each sample point
        """
        for p in self.points:
            p.label = self.label(p)

    def center(self):
        """
        recalculate each center according to sample points
        :return:
        """
        try:
            for clabel in self.centers.iterkeys():
                cps = np.array([p.point for p in self.points if clabel == p.label])
                cps_sum = cps.sum(0)
                new_center = cps_sum * 1.0 / cps_sum.shape[0]
                self.centers[clabel] = LabelPoint(new_center, clabel)
        except Exception, e:
            while True:
                self.init_centers()
                if self.total_center_dist()>0.001:
                    break


    def total_center_dist(self):

        return sum([self.dist(self.centers[label], self.old_centers[label]) for label in self.centers.keys()])

    def dist(self, p1, p2):
        """
        :param p1: LabelPoint 1
        :param p2: LabelPoint 2
        :return: distance between p1 and p2
        """
        return np.sqrt(np.sum((p1.point - p2.point) ** 2))

    def result(self):
        return self.points


def test():
    kmeans = Kmeans(4)
    p1 = [1, 1, 1, 1]
    p2 = [1, 2, 3, 4]
    p3 = [3, 4, 5, 6]
    p4 = [7, 8, 9, 10]
    p5 = [11, 11, 34, 89]
    p6 = [23, 33, 12, 89]
    kmeans.set_sample([p1, p2, p3, p4, p5, p6])
    kmeans.train()
    print "Center:"
    for p in kmeans.centers.itervalues():
        print p
    print "Result:"
    for p in kmeans.result():
        print p


if __name__ == '__main__':
    test()
