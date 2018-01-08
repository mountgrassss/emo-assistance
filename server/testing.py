import numpy as np
from keras import layers
#from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
#from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import sys

def test(img_path):
    happyModel = load_model('happy_model.h5')

    ### START CODE HERE ###
    #img_path = 'images/mytest.jpg'
    ### END CODE HERE ###
    img = image.load_img(img_path, target_size=(64, 64))
    #imshow(img)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    return (happyModel.predict(x))

program_name = sys.argv[0]
image_path = sys.argv[1]
test(image_path)