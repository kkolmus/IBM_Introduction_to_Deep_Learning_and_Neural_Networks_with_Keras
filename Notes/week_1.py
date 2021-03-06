# Artificial Neural Networks - Forward Propagation

# 1. LIBS 

# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g. your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.21.4

# 2. Import Numpy library to generate weights and biases

import numpy as np 

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

print(weights)
print(biases)

# 3. Let's compute the output for a given input

x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

# 4. Computing the wighted sum of the inputs at the first node of the hidden layer.

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_12))

print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

# 5. Compute the activation of the second node in the hidden layer.

a_12 = 1.0 / (1.0 + np.exp(-z_12))

print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

# 6. Compute the weighted sum of these inputs to the node in the output layer

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]

print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

a_2 = 1.0 / (1.0 + np.exp(-z_2))

print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))



# Obviously, neural networks for real problems are composed of many hidden layers 
# and many more nodes in each layer. So, we can't continue making predictions 
# using this very inefficient approach of computing the weighted sum at each node 
# and the activation of each node manually.

# In order to code an automatic way of making predictions,
# let's generalize our network. A general network would take $n$ inputs, 
# would have many hidden layers, each hidden layer having $m$ nodes, and would have an output layer. 
# Although the network is showing one hidden layer, 
# but we will code the network to have many hidden layers. 
# Similarly, although the network shows an output layer with one node, 
# we will code the network to have more than one node in the output layer.

# 6. Initialize a Network

n = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

# 7. Inititailize the weights and the biases in the network to random numbers. 
# In order to be able to initialize the weights and the biases to random numbers,
# we will need to import the **Numpy** library.

import numpy as np # import the Numpy library

num_nodes_previous = n # number of nodes in the previous layer

network = {} # initialize network an an empty dictionary

# loop through each layer and randomly initialize the weights and biases associated with each node
# notice how we are adding 1 to the number of hidden layers in order to include the output layer
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network


#### Use the *initialize_network* function to create a network that:
# 
# 1.  takes 5 inputs
# 2.  has three hidden layers
# 3.  has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# 4.  has 1 node in the output layer
#
# Call the network **small_network**.

small_network = initialize_network(
  num_inputs = 5, 
  num_hidden_layers = 3, 
  num_nodes_hidden = [3, 2, 3], 
  num_nodes_output = 1)


# 8. Compute Weighted Sum at Each Node

# The weighted sum at each node is computed as the dot product of the inputs 
# and the weights plus the bias. 
# So let's create a function called *compute_weighted_sum* that does just that.

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias
  
# Let's generate 5 inputs that we can feed to **small_network**.

from random import seed
import numpy as np

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(inputs))

# Use the *compute_weighted_sum* function to 
# compute the weighted sum at the first node in the first hidden layer.

node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(
  inputs = inputs, 
  weights = node_weights, 
  bias = node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))


# 9. Compute Node Activation

# Recall that the output of each node is simply a non-linear tranformation of the weighted sum. 
# We use activation functions for this mapping. 
# Let's use the sigmoid function as the activation function here. 
# So let's define a function that takes a weighted sum as input and 
# returns the non-linear transformation of the input using the sigmoid function.

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# Use the *node_activation* function to compute the output of the first node 
# in the first hidden layer.

node_output  = node_activation(
  weighted_sum = compute_weighted_sum(
    inputs = inputs, 
    weights = node_weights, 
    bias = node_bias))
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))


# 10. Forward Propagation

# The final piece of building a neural network that can perform predictions 
# is to put everything together. 
# So let's create a function that applies 
# the *compute_weighted_sum* and *node_activation* functions 
# to each node in the network and propagates the data all the way 
# to the output layer and outputs a prediction for each node in the output layer.

# The way we are going to accomplish this is through the following procedure:
# 
# 1.  Start with the input layer as the input to the first hidden layer.
# 2.  Compute the weighted sum at the nodes of the current layer.
# 3.  Compute the output of the nodes of the current layer.
# 4.  Set the output of the current layer to be the input to the next layer.
# 5.  Move to the next layer in the network.
# 6.  Repeat steps 2 - 4 until we compute the output of the output layer.

def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions
  

# Use the *forward_propagate* function to compute the prediction of our small network

predictions = forward_propagate(
  network = small_network, 
  inputs = inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))

# So we built the code to define a neural network. 
# We can specify the number of inputs that a neural network can take, 
# the number of hidden layers as well as the number of nodes in each hidden layer, 
# and the number of nodes in the output layer.

my_network = initialize_network(
  num_inputs = 5, 
  num_hidden_layers = 3, 
  num_nodes_hidden = [2, 3, 2], 
  num_nodes_output = 3)

# Give an input

inputs = np.around(np.random.uniform(size=5), decimals=2)

predictions = forward_propagate(
  network = my_network, 
  inputs = inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))


