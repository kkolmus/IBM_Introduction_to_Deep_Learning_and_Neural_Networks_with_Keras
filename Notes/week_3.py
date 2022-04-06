# KERAS

# !pip install numpy==1.21.4
# !pip install pandas==1.3.4
# !pip install tensorflow==2.2.0
# !pip install keras==2.1.6
# !pip install matplotlib==3.5.0

# Regression Models with Keras
# number of input neurons is dependent on the number of feature that are used to predict target variable
# output neuron is dependent on the target we wish to predict

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

n_cols = dataset.shape[1]

model.add(Dense(5, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)

model.fit(predictors, target, epochs = 10)

# Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.

model.predict(test_data)


####################
## LAB NOTES ---- 

import pandas as pd
import numpy as np

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

concrete_data.shape

# check for missing values
concrete_data.describe()

concrete_data.isnull().sum()

# Split data into predictors and target

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


predictors.head()
target.head()

# Normalize the data by substracting the mean and dividing by the standard deviation.

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

# Let's save the number of predictors to *n_cols* 
# since we will need this number when building our network.
n_cols = predictors_norm.shape[1] # number of predictors

# Keras runs on top of a low-level library such as TensorFlow. 
# This means that to be able to use the Keras library, 
# you will have to install TensorFlow first 
# and when you import the Keras library, 
# it will be explicitly displayed what backend was used to install the Keras library. In CC Labs, we used TensorFlow as the backend to install Keras, so it should clearly print that when we import Keras.

import keras

# Import the rest of the packages from the Keras library 
# that we will need to build our regressoin model.

from keras.models import Sequential
from keras.layers import Dense

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

## Train and Test the Network

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Other functions that you can use for prediction or evaluation
# https://keras.io/api/models/sequential/


# Classification Models with Keras
# number of input neurons is dependent on the number of feature that are used to predict target variable
# output neuron is dependent on the target we wish to predict
# here: if we want to predict 4 categories we have 4 neurons at the end

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

model = Sequential()

n_cols = dataset.shape[1]

taget = to_categorical(target)

model.add(Dense(5, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))

model.compile(
  optimizer = 'adam',
  loss = 'categorical_crissentropy',
  metrics = ['accuracy'] # you can define your own metric
)

model.fit(predictors, target, epochs = 10)

model.predict(test_data)

####################
## LAB NOTES ---- 

import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# Load the MNIST dataset from the Keras library. 
# The dataset is readily divided into a training set and a test set.

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

# Visualize the first image in the training set using Matplotlib's scripting layer.

plt.imshow(X_train[0])

# With conventional neural networks, 
# we cannot feed in the image as input as is. 
# So we need to flatten the images into one-dimensional vectors, 
# each of size 1 x (28 x 28) = 1 x 784.

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# Since pixel values can range from 0 to 255, 
# normalize the vectors to be between 0 and 1.

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# For classification we need to divide our target variable into categories. 
# We use the to_categorical function from the Keras Utilities package.

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)

# Build a Neural Network

# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
  
  
# Train and Test the Network

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)


# Print the accuracy and the corresponding error.

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# Save model

model.save('classification_model.h5')

# Load model again

from keras.models import load_model

pretrained_model = load_model('classification_model.h5')





