# -*- coding: utf-8 -*-
# Author Wajeeh Ahmed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
%matplitlib inline

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'E:/Stay Away You!/DL_training/Mobile_Net/data/Mobilenet_sample/'
    img = image.load_img(img_path + file,target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_expand_dimension = np.expand_dims(img_array,axis = 0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expand_dimension)    

from IPython.display import Image
Image(filename = 'E:/Stay Away You!/DL_training/Mobile_Net/data/Mobilenet_sample/1.jpg',width= 300,height=200)

preprocessed_img = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_img)
results = imagenet_utils.decode_predictions(predictions)
results

from IPython.display import Image
Image(filename = 'E:/Stay Away You!/DL_training/Mobile_Net/data/Mobilenet_sample/3.jpg',width= 300,height=200)


preprocessed_img = prepare_image('3.jpg')
predictions = mobile.predict(preprocessed_img)
results = imagenet_utils.decode_predictions(predictions)
results
