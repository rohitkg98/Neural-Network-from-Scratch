
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets
import torch
import torchvision
from torchvision import datasets as dset
from torchvision import transforms as transforms

# TODO: Defining torchvision transforms for preprocessing
# TODO: Using torchvision datasets to load MNIST
# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = dset.MNIST(root="./", train=True, transform=trans, download=True)
test_set = dset.MNIST(root="./", train=False, transform=trans, download=True)

batch_size = 4

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 3 # Or keep it as is


# TODO: Training and Validation Loop
# >>> for n epochs
for epoch in range(N_epoch):
    ## >>> for all mini batches
    for batch_index, (inputs, labels) in enumerate(train_loader):
        ### >>> net.train(...)
        print("training iteration no: %d" % batch_index)
        net.train(torch.reshape(inputs, (batch_size, N_in)), labels, lr, debug=True)
    ## at the end of each training epoch, shuffling data for evaluation
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    ## >>> net.eval(...) to check if model is overfitting
    for batch_index, (inputs, labels) in enumerate(train_loader):
        print("validation iteration no: %d" % batch_index)
        net.eval(torch.reshape(inputs, (batch_size, N_in)), labels, debug=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)
prediction_accuracy = []
for batch_index, (inputs, labels) in enumerate(test_loader):
    #result are the predictions
    score, result = net.predict(torch.reshape(inputs, (batch_size, N_in)))
    correct = torch.eq(result, labels)
    accuracy = float(sum(correct)/len(labels))
    prediction_accuracy.append(accuracy)

#to find accuracy of the trained model
final_accuracy = sum(prediction_accuracy)/len(labels)
print("Accuracy of the trained model on MNIST Dataset is %f %%" % (final_accuracy*100))
