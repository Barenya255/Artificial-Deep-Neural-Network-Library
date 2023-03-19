# CS6910Assignment1

# Q1

- log down 10 unique images corresponding to 10 different labels.
- class PreProc() contains the required functions to get this and other preprocessing done
- cv2 used to add Gaussian Blur to the images(better generalisation)

# Q2

- classes Algorithms introduced - containes ForwardProp() which can be used to attain the current prediction
- class FFNet() has functions to create and maintain a neural network using numpy arrays
     - addHiddenLayer(number of neurons, type of initialization) is used to add a hidden layer of specified width at the end of the current neural Netowrk
     - addOutputLayer(number of classes, type of initializaion) is used to add the output Layer at the end of the neural Netwrok
     - NOTE: care should be taken not to add another hiddenLayer after the outputLayer. Order of the statements is important.
- class Functions() has all functions as staticmethods which can be called upon whenever required.

## Observation:

- only 10% accuracy without any training is achieved as all weights are intialized somewhat randomly with respect to each other.
- Therefore all inputs are more or less predicting the same output.
- Thus only 10% accuracy

# Q3

- Introduced BackProp() to class Algorithms. It basically gives us the derivative of the entire neural Network with respect to an input
- Introduced the following optimizers to class Algorithms
    - miniBatchGD(...) = sgd
    - miniBatchMGD(...) = momentum based sgd
    - miniBatchNAG(...) = nesterov accelerated gd
    - rmsprop(...) = RMS PROP
    - adam(...)
    - nadam(...)
 - Introduced fit(...) to 
