# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        for i in range(len(data_files)):
            
            dict = unpickle(data_files[i])
            # corpus = [(data[i], labels[i]) for i in range(len(labels))]
            if i == 0:
                self.data = dict[b"data"]
                self.labels = dict[b"labels"]
            else:
                self.data += dict[b"data"]
                self.labels += dict[b"labels"]
        self.transform = transform
        self.target_transform = target_transform
        
        #raise NotImplementedError("You need to write this part!")

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.labels)
        #raise NotImplementedError("You need to write this part!")

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 

        Parameters:
            x:      an integer, used to index into your data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        image = self.data[idx]
        label = self.labels[idx]
        image = np.reshape(image, (3, 32, 32))
        image = np.transpose(image, (1,2,0))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
            
        #raise NotImplementedError("You need to write this part!")
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    
    return transforms.Compose([transforms.ToTensor()])

    #raise NotImplementedError("You need to write this part!")


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    return CIFAR10(data_files, transform=transform)
    
    
    
    
    


"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
        
    """
    return DataLoader(dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle'])
    # raise NotImplementedError("You need to write this part!")


"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################
        # 1. Initialize convolutional backbone with pretrained model parameters.
        pretrained_model = resnet18()
        pretrained_model.load_state_dict(torch.load("resnet18.pt"))

        # 2. Freeze convolutional backbone.
        for param in pretrained_model.parameters():
            param.requires_grad = False
        # 3. Initialize linear layer(s). 
        pretrained_model.fc = nn.Linear(512, 8)
        self.pre_model = pretrained_model
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        # resize / reshape x to ResNet18 
        return self.pre_model(x)
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if(optim_type == "Adam"):
        return torch.optim.Adam(model_params, lr=0.0448)
    else:
        return torch.optim.SGD(model_params, lr=0.0448)
    #raise NotImplementedError("You need to write this part!")


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """
    total = 40000
    current = 0
    ################# Your Code Starts Here #################
    for features, labels in train_dataloader:
        y_pred = model(features)
        loss = loss_fn(y_pred, labels)
        current += y_pred.size(0)
        print("loss: ", loss, "ratio: ", current, "/", total)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    # test_loss = something
    # print("Test loss:", test_loss)
    raise NotImplementedError("You need to write this part!")

"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    
    file_path = ["cifar10_batches/data_batch_1", "cifar10_batches/data_batch_2", "cifar10_batches/data_batch_3", "cifar10_batches/data_batch_4", "cifar10_batches/data_batch_5"]  
    train_data = build_dataset(file_path, get_preprocess_transform(None))
    train_param = {"batch_size": 100, "shuffle": True}
    train_loader = build_dataloader(train_data, train_param)
    
    
    model = FinetuneNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.parameters(), None)
    
    
    for epoch in range(1):
        print("Epoch #", epoch)
        train(train_loader, model, loss_fn, optimizer)  # You need to write this function.
    
    return model
    #raise NotImplementedError("You need to write this part!")
    
