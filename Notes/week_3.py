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
