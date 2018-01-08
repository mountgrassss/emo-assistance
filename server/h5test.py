# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:54:06 2018

@author: fuya
"""

import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio

def image_load(filename):
    im = imageio.imread(filename)
    imgDim = im.shape[0]*im.shape[1]*im.shape[2]
    #print(imgDim)
    im = np.reshape(im, (imgDim,) )
    print(np.shape(im))
    return im

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def create_dataset():
    h5f = h5py.File("datasets/pending_face_test.h5", "w")
    
    image_data = image_load("datasets/happy_2.jpg")
    print(np.shape(image_data))
    img_label = 1
    
    dset_x = h5f.create_dataset("pending_set_x", data = image_data )
    dset_y = h5f.create_dataset("pending_set_y", (1, 100), dtype='i') 
    
    test_x = h5f["pending_set_x"]
    test_y = h5f["pending_set_y"]
    #test_x[0] = image_data
    #test_y[0] = img_label
    print(np.shape(test_x))
    print(np.shape(test_y))
    print("create h5 file")
    print(h5f.keys() )
    h5f.close()
    
def append_image(filename, label):
    image_data = image_load("datasets/happy_2.jpg")
    print(np.shape(image_data))
    
    h5f = h5py.File('datasets/pending_face_test.h5', "a")    
    
    dset_x = h5f["pending_set_x"]
    dset_y = h5f["pending_set_y"]
    # Save data string converted as a np array
    dset_x[:,1] = image_data
    dset_y[0,1] = label
    print(np.shape(dset_x))
    print(np.shape(dset_y))
    print(h5f.keys() )
    
    train_set_x = np.array(dset_x[:,1])
    train_image_x = train_set_x.reshape(1, train_set_x.shape[0])
    print(np.shape(train_image_x) )
    
    h5f.close()

create_dataset()
#append_image('datasets/happy_3.jpg', 1)
#img = image_load('datasets/happy_3.jpg')
