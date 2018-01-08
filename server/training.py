import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import h5py

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().magic('matplotlib inline')


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    X_Input = Input(input_shape)
    
    X = ZeroPadding2D((2,2))(X_Input)  #
    X = Conv2D(32, (5,5), strides=(1,1), name='conv0')(X)  #64*64*32
    X = BatchNormalization(axis = 3, name ='bn0')(X) #64*64*32
    X = Activation('relu')(X)  #64*64*32 
    
    X = MaxPooling2D((2,2), name='max_pool')(X)  #32*32*32
    #####################################################
    
    #X = ZeroPadding2D((1,1))(X)
    #X = Conv2D(32, (3,3), strides=(1,1), name='conv1')(X)
    #X = BatchNormalization(axis = 3, name ='bn1')(X)
    #X = Activation('relu')(X)
    
    #####################################################
    
    X = Flatten()(X)
    X = Dense(1, activation = 'sigmoid', name = 'fc')(X)
    
    model = Model(inputs=X_Input, outputs = X, name='HappyModel')
    ### END CODE HERE ###
    
    return model
    
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#define the model
happyModel = HappyModel(X_train.shape[1:])
happyModel.compile(loss='binary_crossentropy', optimizer='adam', metrics = ["accuracy"])
happyModel.fit(x = X_train, y = Y_train, epochs = 20, batch_size = 32)

### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

happyModel.save('happy_model.h5')