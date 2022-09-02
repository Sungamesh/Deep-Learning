# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 04:41:09 2022

@author: sunga
"""

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.ResNet50 import resnet

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

import tensorflow as tf

import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
import dataset


#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
#from tensorflow import set_random_seed
tf.random.set_seed(2)

#os.system('spd-say -t male3 "I will try to learn this, my master."')

batch_size = 32

#Prepare input data
classes = os.listdir('training_data')
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='training_data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print(len(data.train.load_data()))

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

# add preprocessing layer to the front of VGG
rs = tf.keras.applications.EfficientNetV2B1(input_shape=[img_size,img_size] + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in rs.layers:
  layer.trainable = False
  
# our layers - you can add more if you want
x = Flatten()(rs.output)

pred = Dense(num_classes, activation='softmax')(x)

# create a model object
model = Model(inputs=rs.input, outputs=pred)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

x_train, y_train = data.train.load_data()
x_val, y_val = data.valid.load_data()

# fit the model
r=model.fit(x_train, y_train,
            batch_size = 28,
                         epochs = 25,
                         validation_data = (x_val, y_val))

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()

