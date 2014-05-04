# -*- coding: utf-8 -*-
"""
Created on Sun May 04 16:22:08 2014

@author: break
"""
import os
import matplotlib.image as mpimage
import random
import numpy as np
import gzip
import cPickle
import theano
import theano.tensor as T

def makeFaceVerifyDataSet(base):
    """
    make a face dataset used in face Valid with pcnn
    data:image pairs 
    target:labels of the pairs whether similar of unsimilar
    """
    images = []
    labels = []
    if os.path.isdir(base):
        for i in os.listdir(base):
            curpath = os.path.join(base, i)
            for j in os.listdir(curpath):
                ifile = os.path.join(curpath, j)
                image = mpimage.imread(ifile)
                images.append(image.flatten())
                labels.append(int(i))
    length = len(labels)
    
    #I want to extract the pairs of face
    images_new = []
    labels_new = []
    #extract the similar pairs
    for i in range(length):
        for j in range(i+1,length):
            if labels[i] == labels[j]:
                image_pair = []
                image_pair.append(images[i])
                image_pair.append(images[j])
                images_new.append(image_pair)
                labels_new.append(1)
            else:
                break;
    print("similar pairs:%d" % len(labels_new))
           
    #extract the unsimilar pairs about 2000 numbers
    count = 0
    n_count = len(labels_new)
    idx_tmp = []
    while(1):
        i = np.random.randint(0, length-1)
        j = np.random.randint(i, length-1)
        if labels[i] != labels[j]:
            idx_tmp.append((i,j))
            count += 1
        if count == n_count:
            break;
    idx_tmp = set(idx_tmp)
    print("unsimilar pairs:%d" % len(idx_tmp))
    for i,j in idx_tmp:
        images_new.append([images[i],images[j]])
        labels_new.append(0)
    
    #rerandom the sort of the dataset
    length_new = len(labels_new)
    for k in range(length_new):
        r1 = random.randint(0,length_new-1)
        r2 = random.randint(0,length_new-1)
        if(r1 != r2):
            pair = images_new[r1]
            images_new[r1] = images_new[r2]
            images_new[r2] = pair
            label = labels_new[r1]
            labels_new[r1] = labels_new[r2]
            labels_new[r2] = label
            
    f = gzip.open('face_data_pcnn.pkl.gz', 'wb')
    cPickle.dump((images_new, labels_new), f)
    f.close()

def loadFaceVerifyDataSet(filename):
    f = gzip.open(filename, 'rb')
    images, labels = cPickle.load(f)
    f.close()
    shared_x = theano.shared(np.asarray(images, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.asarray(labels, dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')

if __name__ == "__main__":
    #makeFaceVerifyDataSet('F:\\face\\gray\\')
    x, y = loadFaceVerifyDataSet('face_data_pcnn.pkl.gz')
    print x.get_value(borrow=True)