#!-*- coding: utf8 -*-
import random
import os
from numpy import *
from scipy.misc import imread
from scipy.misc import imresize

class Dataset(object):
    def __init__(self, args={}):
        self.train = []
        self.test = []
        self.files = []
        self.gts = []
        self.gt_train = []
        self.gt_test = []
        self.cross_validation = False
        if (args.has_key('n_folds')):
            self.cross_validation = True
            self.n_folds = args['n_folds']
            self.seed = args['seed']
            self.fold = args['fold']

        self.counter_train = 0
        self.counter_test = 0
        self.file_names()

    def file_names(self):
        files = os.listdir("data/apples/Glomerella")
        self.files += ["Glomerella/" + f for f in files]
        self.gts += [0]*len(files)

        files = os.listdir("data/apples/Herbicida")
        self.files += ["Herbicida/" + f for f in files]
        self.gts += [1]*len(files)

        files = os.listdir("data/apples/Magnesio")
        self.files += ["Magnesio/" + f for f in files]
        self.gts += [2]*len(files)

        files = os.listdir("data/apples/Potassio")
        self.files += ["Potassio/" + f for f in files]
        self.gts += [3]*len(files)

        files = os.listdir("data/apples/Sarna")
        self.files += ["Sarna/" + f for f in files]
        self.gts += [4]*len(files)

        aux = zip(self.files, self.gts)
        try:
            random.seed(self.seed)
        except:
            random.seed()
        random.shuffle(aux)
        im, gt = zip(*aux)
        im, gt = list(im), list(gt)

        if self.cross_validation:
            print("Doing cross validation with: ")
            print("n_folds: ", self.n_folds)
            print("fold: ", self.fold)
            print("seed: ", self.seed)
            images_per_fold = int(len(im)/self.n_folds)
            self.test = im[images_per_fold*self.fold:images_per_fold*(self.fold+1)]
            del im[images_per_fold*self.fold:images_per_fold*(self.fold+1)]
            self.gt_test = gt[images_per_fold*self.fold:images_per_fold*(self.fold+1)]
            del gt[images_per_fold*self.fold:images_per_fold*(self.fold+1)]

            self.train = im
            self.gt_train = gt

        else:
            #70 train 30 test
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
            rgb = imresize((imread('data/apples/' + folder)[:,:]).astype(float32), (227, 227, 3))
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
            rgb = imresize((imread('data/apples/' + folder)[:,:]).astype(float32), (227, 227, 3))
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
