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
