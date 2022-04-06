# Week 2

Vanishing gradiant
W czasie backpropagation,
neurony na wczesnych etapach ucza sie wolniej niz neurony na poznych etapach.
Powod: sigmoid function
Wzor powoduje ze aktywacja ma coraz to mniejsze wartosci, ktore zblizaja sie do 0 na
poznych etapach.

Types of activation Functions

1. the binary step function
2. the linear or identity function
3. the sigmoid or logistic function
4. the hyperbolic tangent, or tanh, function, 
5. the rectified linear unit (ReLU) function, 
6. the leaky ReLU function, 
7. the softmax function.



sigmoid function
At z = 0, a is equal to 0.5 
and when z is a very large positive number, a is close to 1, 
and when z is a very large negative number, a is close to zero.

Another problem with the sigmoid function is that the values only range from 0 to 1. 
This means that the sigmoid function is not symmetric around the origin. 
The values received are all positive. 



hyperbolic tangent function
a scaled version of the sigmoid function
symertric over origin
still the vanishing gradient problem



ReLU function
non-linear
it does not activate all the neurons at the same time
sparse activation of neurons overcomes the vanishing gradient problem
 


softmax function
a type of a sigmoid function
handy when we are trying to handle classification problems.
The softmax function is ideally used in the output layer of the classifier 
where we are actually trying to get the probabilities to define the class of each input.



