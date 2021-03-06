# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
%matplitlib inline

# Model fine tuning layer adjustment convert from 1000 class classification to 2 class classification
model = tf.keras.applications.vgg16.VGG16()
model_ = Sequential()
for layers in model.layers[:-1]:
    model_.add(layers)
model_.summary()

for layers in model_.layers:
    layers.trainable=False
    
model_.add(Dense(units=2,activation='softmax'))
model_.summary()

train_path = 'E:/Stay Away You!/DL_training/CNN/data/dogs-vs-cats/train'
valid_path = 'E:/Stay Away You!/DL_training/CNN/data/dogs-vs-cats/valid'
test_path = 'E:/Stay Away You!/DL_training/CNN/data/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory= train_path,target_size=(224,224),classes=['cat','dog'],batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory= valid_path,target_size=(224,224),classes=['cat','dog'],batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory= test_path,target_size=(224,224),classes=['cat','dog'],batch_size=10,shuffle=False)

#Model Compilation
model_.compile(optimizer=Adam(learning_rate = 0.0001),loss = 'categorical_crossentropy',metrics=['accuracy'])
model_.fit(x=train_batches,validation_data=valid_batches,epochs=5,verbose = 2)

#Prediction
predictions = model_.predict(x=test_batches,verbose=0)

cm = confusion_matrix( y_true = test_batches.classes, y_pred = np.argmax(predictions,axis=-1))
test_batches.class_indices

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Metrix')