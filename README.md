# CS6910Assignment1

# Instructions for using train.py

- note: once python train.py is run from the directory, it will ask for the login key. On giving it an appropriate key, it will start the run
- no need to import anything.
- just add the key for user's wandb to login.
- "python train.py" to log into wandb the values
- It would also print out the accuracy on the test set once the training is over.
- All libraries kept in same file to avoid dependency issues.

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
 - Introduced fit(...) to class FFNet() for training the neural network
 - Used Functions.plot() to plot the loss, validation loss, validation accuracy and accuracy per epoch of the training process
 
# Q4
 
 - Introduced wandb.log() to log files at wandb, for more details

# Q5

- wandb plot asked for, please refer to the report for the same. 

# Q6

- parallel coordinates plot generated by wandb and reports to be referred to for the same.

# Q7

- To form confusion matrix, appropriate wandb logging function used. This seams to be more informative than the one provided by sklearn

# Q8
- modified the BackProp() in class Algorithms to support for Mean Squared Error
- added extra parameter for selecting appropriate loss function so that crossEntropy and mse can be switched between
- added MSE Funciton to Functions class
- added if else selector to Loss calculation so that loss can be selected

# Q9
- link to this repository has been provided

# Q10
- Used similar parameters as the fashion minst model, added support for choosing between mnist and fashion mnist datasets

