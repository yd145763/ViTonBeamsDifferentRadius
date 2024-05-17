# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:53:22 2024

@author: limyu
"""

import numpy as np


import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import keras
import pandas as pd 
import statistics

os.environ['KERAS_BACKEND'] = 'tensorflow' 
num_head_list = []
num_trans_list = []
mlp_size_list = []
accuracy_list = []
training_loss_list = []
validation_loss_list = []
time_list = []
confusion_matrix_list = []


from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential



SIZE = 64

    
    
    
image_directory = 'C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\MixedPitchDifferentRadius\\'
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

reactive_images = os.listdir(image_directory + 'reactive_focused\\')
for i, image_name in enumerate(reactive_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'reactive_focused\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

nearfield_images = os.listdir(image_directory + 'nearfield_focused\\')
for i, image_name in enumerate(nearfield_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'nearfield_focused\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

farfield_images = os.listdir(image_directory + 'farfield_focused\\')
for i, image_name in enumerate(farfield_images):
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'farfield_focused\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(2)

sparsed_reactive_images = os.listdir(image_directory + 'reactive_sparsed\\')
for i, image_name in enumerate(sparsed_reactive_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'reactive_sparsed\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(3)

sparsed_nearfield_images = os.listdir(image_directory + 'nearfield_sparsed\\')
for i, image_name in enumerate(sparsed_nearfield_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'nearfield_sparsed\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(4)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

sparsed_farfield_images = os.listdir(image_directory + 'farfield_sparsed\\')
for i, image_name in enumerate(sparsed_farfield_images):
    if (image_name.split('.')[2] == 'png'):
        image = cv2.imread(image_directory + 'farfield_sparsed\\' + image_name)
        print(image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(5)


CONFIGURATION = {
    "BATCH_SIZE": 16,
    "IM_SIZE": 64,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 5,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 256,
    "NUM_CLASSES": 6,
    "PATCH_SIZE": 8,
    "PROJ_DIM": 192,
    "CLASS_NAMES": ["reactive", "nearfield", "farfield"],
}

import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import time
import random
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature

class PatchEncoder(Layer):
  def __init__(self, N_PATCHES, HIDDEN_SIZE):
    super(PatchEncoder, self).__init__(name = 'patch_encoder')

    self.linear_projection = Dense(HIDDEN_SIZE)
    self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE )
    self.N_PATCHES = N_PATCHES

  def call(self, x):
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    patches = tf.reshape(patches, (tf.shape(patches)[0], CONFIGURATION["PATCH_SIZE"]**2, patches.shape[-1]))

    embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1 )
    output = self.linear_projection(patches) + self.positional_embedding(embedding_input)

    return output

class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name = 'transformer_encoder')

    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()

    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE )

    self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)

  def call(self, input):
    x_1 = self.layer_norm_1(input)
    x_1 = self.multi_head_att(x_1, x_1)

    x_1 = Add()([x_1, input])

    x_2 = self.layer_norm_2(x_1)
    x_2 = self.dense_1(x_2)
    output = self.dense_2(x_2)
    output = Add()([output, x_1])

    return output

class ViT(Model):
  def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
    super(ViT, self).__init__(name = 'vision_transformer')
    self.N_LAYERS = N_LAYERS
    self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
    self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
    self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = 'softmax')
  def call(self, input, training = True):

    x = self.patch_encoder(input)

    for i in range(self.N_LAYERS):
      x = self.trans_encoders[i](x)
    x = Flatten()(x)
    x = self.dense_1(x)
    x = self.dense_2(x)

    return self.dense_3(x)

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',  # Monitor training loss
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=100,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode
    mode='min',  # Minimize the monitored quantity
    restore_best_weights=True  # Whether to restore model weights to the best observed during training
)

N_HEADS = 1
N_LAYERS = 1
N_DENSE_UNITS = 128


vit = ViT(N_HEADS = N_HEADS, HIDDEN_SIZE = (CONFIGURATION["PATCH_SIZE"]**2)*3, N_PATCHES = CONFIGURATION["PATCH_SIZE"]**2,
    N_LAYERS =N_LAYERS, N_DENSE_UNITS = N_DENSE_UNITS)
vit(tf.zeros([2,CONFIGURATION["PATCH_SIZE"]**2,CONFIGURATION["PATCH_SIZE"]**2,3]))
vit.summary()
vit.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.40, random_state = 0)


# ### Training the model
# As the training data is now ready, I will use it to train the model.   




#Fit the model
history = vit.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 5, 
                         verbose = 1, 
                         epochs = 100,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.5,
                         shuffle = True,
                         callbacks=[early_stopping_callback]
                     )


"""
image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\MixedPitchDifferentRadius\\farfield_sparsed\\grating012umpitch05dutycycle30um_37.72.png")
image = Image.fromarray(image, 'RGB')
image = image.resize((SIZE, SIZE))
img = np.array(image)

# Add batch dimension to the input image
img_with_batch = tf.expand_dims(img, axis=0)


# Visualize the image after the patch encoder
output_after_patch_encoder = vit.layers[0](img_with_batch) #patch encoder by layer index
output_array = output_after_patch_encoder.numpy()
plt.imshow(output_array[0], cmap = 'turbo')  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

# Visualize the image after each Transformer encoder

output_after_transformer0 = vit.layers[1](output_after_patch_encoder)
output_array = output_after_transformer0.numpy()
plt.imshow(output_array[0], cmap = 'turbo')  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

output_after_transformer1 = vit.layers[2](output_after_transformer0)
output_array = output_after_transformer1.numpy()
plt.imshow(output_array[0], cmap = 'rainbow')  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

output_after_transformer2 = vit.layers[3](output_after_transformer1)
output_array = output_after_transformer2.numpy()
plt.imshow(output_array[0], cmap = 'rainbow')  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

output_after_transformer2 = np.array(output_after_transformer2)
output_after_transformer2 = np.tile(output_after_transformer2, (1, 1, 1, 1))
output_after_transformer2 = output_after_transformer2.reshape((1, -1))

n = np.arange(0, len(output_after_transformer2.transpose()), 1)
plt.scatter(n, output_after_transformer2.transpose(), s=0.1)  
plt.show()


output_after_dense0 = vit.layers[4](output_after_transformer2)
output_array = output_after_dense0.numpy()
output_array = output_array.transpose() 
n = np.arange(0, len(output_array), 1)
plt.scatter(n, output_array, s=5)  
plt.show()

output_after_dense1 = vit.layers[5](output_after_dense0)
output_array = output_after_dense1.numpy()
output_array = output_array.transpose()
n = np.arange(0, len(output_array), 1) 
plt.scatter(n, output_array, s=5)  
plt.show()

output_after_dense2 = vit.layers[6](output_after_dense1)
output_array = output_after_dense2.numpy()
output_array = output_array.transpose()
n = np.arange(0, len(output_array), 1) 
plt.scatter(n, output_array, s=30)  
plt.show()
"""
#================================================================

from matplotlib.ticker import StrMethodFormatter
import numpy as np
from matplotlib.colors import LogNorm

image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\MixedPitchDifferentRadius\\farfield_sparsed\\grating012umpitch05dutycycle30um_37.72.png")
image = cv2.imread("C:\\Users\\limyu\\Google Drive\\CNNBeamProfiles\\MixedPitchDifferentRadius\\farfield_sparsed\\grating012umpitch05dutycycle30um_35.03.png")

image = Image.fromarray(image, 'RGB')
image = image.resize((SIZE, SIZE))
img = np.array(image)




test_image_rgb = tf.reverse(img, axis=[-1])

# Extract patches
patches = tf.image.extract_patches(images=tf.expand_dims(test_image_rgb, axis=0),
                                   sizes=[1, 8, 8, 1],
                                   strides=[1, 8, 8, 1],
                                   rates=[1, 1, 1, 1],
                                   padding='VALID')

print(patches.shape)
patches = tf.reshape(patches, (patches.shape[0], -1, 192))
print(patches.shape)
patches = patches[0,:,:]

before_T = patches

fig = plt.figure(figsize=(18, 4))
ax = plt.axes()
cp=ax.imshow(patches, cmap = 'turbo')
colorbarmax = img.max().max()
clb=fig.colorbar(cp, cmap = 'turbo', pad = 0.01)
clb.ax.set_title('Values', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
#plt.imshow(img[0], cmap = 'turbo', norm=LogNorm())  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

fig = plt.figure(figsize=(6, 6))
ax = plt.axes()
cp=ax.imshow(test_image_rgb, cmap = 'turbo')
colorbarmax = img.max().max()
clb=fig.colorbar(cp, cmap = 'turbo', pad = 0.01)
clb.ax.set_title('Values', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
#plt.imshow(img[0], cmap = 'turbo', norm=LogNorm())  # Assuming batch size is 1
plt.axis('off')
plt.show()
plt.close()

# Add batch dimension to the input image
img = tf.expand_dims(img, axis=0)
I = np.arange(0, len(vit.layers), 1)


for i in I:
    print(vit.layers[i])

data_in_layers = []
for i in I[0:N_LAYERS+1]:
    print(vit.layers[i])
    img = vit.layers[i](img) #patch encoder by layer index
    img = img.numpy()
    data_in_layers.append(img)
    fig = plt.figure(figsize=(18, 4))
    ax = plt.axes()
    cp=ax.imshow(img[0], cmap = 'turbo')
    colorbarmax = img.max().max()
    clb=fig.colorbar(cp, cmap = 'turbo', pad = 0.01)
    clb.ax.set_title('Values', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    #plt.imshow(img[0], cmap = 'turbo', norm=LogNorm())  # Assuming batch size is 1
    plt.axis('off')
    plt.show()
    plt.close()


img = np.tile(img, (1, 1, 1, 1))
img = img.reshape((1, -1))
img = np.array(img)
img = img.transpose()
n = np.arange(0, len(img), 1)
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.scatter(n, img, s = 0.1, color = 'red', alpha = 1)
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Data")
plt.ylabel("Values")
#plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "best")
plt.show()
plt.close()
img = img.transpose()

total_params_model = sum(tf.size(p).numpy() for p in vit.trainable_variables)


meow1 = data_in_layers[3]
meow1 = meow1[0,:,:]
meow2 = data_in_layers[4]
meow2 = meow2[0,:,:]
meow1.shape[0]

data_array = np.zeros((64, 192))
for i in range(meow1.shape[0]):
    for j in range(meow1.shape[1]):
        difference = abs((meow2[i,j] - meow1[i,j])/meow1[i,j])
        data_array[i,j] = difference
print(data_array.mean())

for i in I[2:5]:
    print(vit.layers[i])
    img = vit.layers[i](img) #patch encoder by layer index
    img = img.numpy()
    img = img.transpose()
    n = np.arange(0, len(img), 1)
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    ax.scatter(n, img, s = 50, color = 'red', alpha = 1)
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(15)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel("Data")
    plt.ylabel("Values")
    #plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "best")
    plt.show()
    plt.close()
    img = img.transpose()