import torch, random, math, json
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def train_model():
    model = torch.nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_features=8*8*15, out_features=300),
        nn.ReLU(),
        nn.Linear(in_features=300, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=1)
    )

    # Initialize the weights (if necessary)
    # model[1].weight.data = initialize_weights()
    # model[1].bias.data = torch.zeros(1)

    # Load the dataset and create a data loader
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    # Training loop
    for epoch in range(2000):
        for x, y in trainloader:
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()

    # Save the trained model
    torch.save(model, 'model.pkl')

if __name__ == "__main__":
    train_model()
    
