# Neural-Network-from-Scratch
A Neural Network from scratch with two hidden layers. Initial layer and hidden layers activated by implementing sigmoid function. Final layer activated by implementing softmax function. Backpropagation implemented after deriving the functions on paper.

Utilizing this neural net, to create an MNIST Recognizer with upto 96% accuracy after training on 60,000 images from PyTorch MNIST dataset.

##Directory Contents
*nnet
  contains implementation of the neural network.
  *activation.py
    implementation of activation function and its derivative.
  *loss.py
    implementation of loss function and its derivative.
  *model.py
    implementation of the neural net, providing a foward pass and a backpropagation, along with other useful methods.
  *optimizer.py
    implementation of function to update weights of the neural network.
*main.py
  utilizing the neural net to train an MNIST Recognising Model.
  
The tests are not written by me.
