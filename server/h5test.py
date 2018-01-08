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
    #plt.imshow(im)
    
    return im

def gray_image_load(filename):
    im = imageio.imread(filename)
    plt.imshow(im)
    grayIm = color_to_gray(im)
    
    imgDim = grayIm.shape[0]*grayIm.shape[1]
    plt.imshow(grayIm)
    #print(imgDim)
    reshapedIm = np.reshape(grayIm, (1, imgDim) )
    print(np.shape(reshapedIm))
    return reshapedIm

def average(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def color_to_gray(image):
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array    
    # get row number
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = weightedAverage(image[rownum][colnum])

    return grey

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    print(np.shape(train_set_x_orig) )
    print(np.shape(train_set_y_orig) )

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    print(np.shape(test_set_x_orig) )

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    print(np.shape(train_set_y_orig) )
    print(np.shape(test_set_y_orig) )
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def copy_dataset():

    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], 1))
    print(np.shape(train_set_x_orig) )
    print(np.shape(train_set_y_orig) )
    train_dataset.close()

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0],1))
    test_dataset.close()
    
    newH5fTrain = h5py.File('datasets/train_happy_chunked.h5', "w")

    newH5fTrain.create_dataset("train_set_x", data = train_set_x_orig, maxshape=(None,64,64,3), chunks=True)
    newH5fTrain.create_dataset("train_set_y", data = train_set_y_orig, maxshape=(None,1), chunks=True) 
    train_x = newH5fTrain["train_set_x"]
    train_y = newH5fTrain["train_set_y"]
    #test_x[0] = image_data
    #test_y[0] = img_label
    print(np.shape(train_x))
    print(np.shape(train_y))
    newH5fTrain.close()
    
    newH5fTest = h5py.File('datasets/test_happy_chunked.h5', "w")
    
    newH5fTest.create_dataset("test_set_x", data = test_set_x_orig, maxshape=(None,64,64,3), chunks=True)
    newH5fTest.create_dataset("test_set_y", data = test_set_y_orig, maxshape=(None,1), chunks=True) 
    test_x = newH5fTest["test_set_x"]
    test_y = newH5fTest["test_set_y"]
    print(np.shape(test_x))
    print(np.shape(test_y))
    #print("create h5 file")
    #print(h5f.keys() )
    newH5fTest.close()
    

def update_dataset():
    pendingh5f = h5py.File("datasets/pending_face_test.h5", "r")
    pending_set_x = pendingh5f["pending_set_x"]
    pending_set_y = pendingh5f["pending_set_y"]
    training_ratio = 0.8
    pending_size = np.shape(pending_set_x)[0]
    training_size = int(training_ratio*pending_size)
    X_train_data = pending_set_x[0:training_size,:,:,:]
    Y_train_data = pending_set_y[0:training_size,:]
    X_test_data = pending_set_x[training_size:,:,:,:]
    Y_test_data = pending_set_y[training_size:,:]
    print(np.shape(X_train_data))
    print(np.shape(X_test_data))
    print(np.shape(Y_train_data))
    print(np.shape(Y_test_data))
    pendingh5f.close()
        

    with h5py.File('datasets/train_happy_chunked.h5', 'a') as h5f:
        h5f["train_set_x"].resize((h5f["train_set_x"].shape[0] + X_train_data.shape[0]), axis = 0)
        h5f["train_set_x"][-X_train_data.shape[0]:] = X_train_data

        h5f["train_set_y"].resize((h5f["train_set_y"].shape[0] + Y_train_data.shape[0]), axis = 0)
        h5f["train_set_y"][-Y_train_data.shape[0]:] = Y_train_data
     
        dset_x = h5f["train_set_x"]
        dset_y = h5f["train_set_y"]
        print(np.shape(dset_x))
        print(np.shape(dset_y))
        
    with h5py.File('datasets/test_happy_chunked.h5', 'a') as h5f:
        h5f["test_set_x"].resize((h5f["test_set_x"].shape[0] + X_test_data.shape[0]), axis = 0)
        h5f["test_set_x"][-X_test_data.shape[0]:] = X_test_data

        h5f["test_set_y"].resize((h5f["test_set_y"].shape[0] + Y_test_data.shape[0]), axis = 0)
        h5f["test_set_y"][-Y_test_data.shape[0]:] = Y_test_data
        dset_x = h5f["test_set_x"]
        dset_y = h5f["test_set_y"]
        print(np.shape(dset_x))
        print(np.shape(dset_y))

def create_dataset(dataSetFilePath):
    h5f = h5py.File(dataSetFilePath, "w")
    
    numFile = 1
    pending_set_x = []
    pending_set_y = []
    for i in range(0,numFile):
        image_data = image_load("images/happy_2.jpg")
        print(np.shape(image_data))
        img_label = [1]
        pending_set_x.append(image_data)
        pending_set_y.append(img_label)
    
    h5f.create_dataset("pending_set_x", data = pending_set_x, maxshape=(None,64,64,3), chunks=True)
    h5f.create_dataset("pending_set_y", data = pending_set_y, maxshape=(None,1), chunks=True) 
    test_x = h5f["pending_set_x"]
    test_y = h5f["pending_set_y"]
    #test_x[0] = image_data
    #test_y[0] = img_label
    print(np.shape(test_x))
    print(np.shape(test_y))
    #print("create h5 file")
    #print(h5f.keys() )
    h5f.close()
    
def append_image(filename, label):
    image_data = image_load(filename)
    print(np.shape(image_data))
    
    with h5py.File('datasets/pending_face_test.h5', 'a') as h5f:
        h5f["pending_set_x"].resize((h5f["pending_set_x"].shape[0] + 1), axis = 0)
        h5f["pending_set_x"][-1:] = image_data

        h5f["pending_set_y"].resize((h5f["pending_set_y"].shape[0] + 1), axis = 0)
        h5f["pending_set_y"][-1:] = [label]
    
        dset_x = h5f["pending_set_x"]
        dset_y = h5f["pending_set_y"]
        print(np.shape(dset_x))
        print(np.shape(dset_y))


#load_dataset()
#create_dataset()
'''
n = 4
for i in range(3,3+n):
    filePath = 'images/happy_'+str(i)+'.jpg'
    print(filePath)
    append_image(filePath, 1)
'''
#copy_dataset()
update_dataset()
