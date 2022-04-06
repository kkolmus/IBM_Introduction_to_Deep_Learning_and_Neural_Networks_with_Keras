# Supervised Depp Learning

# Convolutional NN

import keras

model = Sequential()

input_shape = (128, 128, 3)

model.add(
  Conv2D(
    16, 
    kernel_size = (2,2),
    strides = (1,1),
    activation = 'relu',
    input_shape = input_shape
  )
)

model.add(Conv2D(
  32,
  kernel_size = (2,2),
  activation = 'relu'
))

model.add(Conv2D(
  MaxPooling2D(pool_size = (2,2))
))


model.add(Flatten())

model.add(
  Dense(
    100, 
    activation = 'relu'
))

model.add(
  num_classes, 
  activation = 'softmax'
))


# Recurrent NN

# good for genomics data analysis

Long Short-Term Memory Model (LSTM)



# Unsupervised Deep Learning

# Autoencoders

# Autoencoding is a data compression algorithm 
# where the compression and the decompression functions 
# are learned automatically from data 
# instead of being engineered by a human. 

# It tries to predict X from X without the need foR any labels.

# Autoencoders are built using neural networks. 

# Autoencoders are data specific.

# good for dimensionalality reduction

# A very popular type of autoencoders is the 
# Restricted Boltzmann Machines or (RBMs).

# They can learn the distribution of the minority class
# in an imbalance dataset, and then generate more data points \
# of that class, transforming the imbalance dataset into a 
# balanced data set. 

# RBMs can also be used to estimate missing values 
# in different features of a data set. 
# Another popular application of Restricted Boltzmann Machines
# is automatic feature extraction of especially unstructured data. 
# And this concludes our high-level introduction to 
# autoencoders and Restricted Boltzmann Machines.

# LIBS and PACKAGES

#!pip install numpy==1.21.4
#!pip install pandas==1.3.4
#!pip install keras==2.1.6

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

########################
## 1. Convolutional Layer with One set of convolutional and pooling layers

# import data
from keras.datasets import mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixels
X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

# Convert target variabe into categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1] # number of categories

# Define a function that creates our model. 
# Start with one set of convolutional and pooling layers.

def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model
  
  
# build the model
model = convolutional_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))


########################
## 2. Convolutional Layer with two sets of convolutional and pooling layers

# Define a function that creates our model. 
# Start with one set of convolutional and pooling layers.

def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model
  
  
# build the model
model = convolutional_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
