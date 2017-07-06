import random
from numpy import *
from scipy.misc import imread
from scipy.misc import imresize

class Dataset(object):
    def __init__(self):
        self.train = []
        self.test = []
        self.files = []
        self.gts = []
        self.gt_train = []
        self.gt_test = []

        self.counter_train = 0
        self.counter_test = 0
        self.file_names()

    def file_names(self):
        for i in range(582):
            self.files.append("Glomerella/glomerella_" + str(i))
            self.gts.append(0)
        for i in range(340):
            self.files.append("Herbicida/herbicida_" + str(i))
            self.gts.append(1)
        for i in range(370):
            self.files.append("Magnesio/magnesio_" + str(i))
            self.gts.append(2)
        for i in range(356):
            self.files.append("Potassio/potassio_" + str(i))
            self.gts.append(3)
        for i in range(408):
            self.files.append("Sarna/sarna_" + str(i))
            self.gts.append(4)

        aux = zip(self.files, self.gts)
        random.seed()
        random.shuffle(aux)
        im, gt = zip(*aux)

        self.train = im[0:int(0.7*len(im))]
        self.gt_train = gt[0:int(0.7*len(im))]

        self.test = im[int(0.7*len(im)):]
        self.gt_test = gt[int(0.7*len(im)):]

    def get_training_size(self):
        return len(self.train)

    def get_test_size(self):
        return len(self.test)

    def next_batch(self, batch_size):
        images = []
        truths = []
        for i in range(0, batch_size):
            folder = self.train[self.counter_train]
            truth = self.gt_train[self.counter_train]
            rgb = imresize((imread('data/apples/' + folder + '.jpg')[:,:]).astype(float32), (227, 227, 3))
            images.append(rgb)
            one_hot = [0.0]*5
            one_hot[truth] = 1.0
            truths.append(one_hot)
            self.counter_train += 1
            if (self.counter_train >= self.get_training_size()):
                aux = zip(self.train, self.gt_train)
                random.shuffle(aux)
                self.train, self.gt_train = zip(*aux)
                self.counter_train = 0

        return images, truths

    def next_test(self, batch_size):
        images = []
        truths = []
        for i in range(0, batch_size):
            folder = self.test[self.counter_test]
            truth = self.gt_test[self.counter_test]
            rgb = imresize((imread('data/apples/' + folder + '.jpg')[:,:]).astype(float32), (227, 227, 3))
            images.append(rgb)
            one_hot = [0.0]*5
            one_hot[truth] = 1.0
            truths.append(one_hot)
            self.counter_test += 1
            if (self.counter_test >= self.get_test_size()):
                aux = zip(self.test, self.gt_test)
                random.shuffle(aux)
                self.test, self.gt_test = zip(*aux)
                self.counter_test = 0

        return images, truths
