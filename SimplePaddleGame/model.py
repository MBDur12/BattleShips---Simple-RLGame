import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LinQNetwork(nn.Module):
    def __init__(self, input_size, hidden_lyr_size, output_size):
        super().__init__() # Instantiates the layers of the NN.
        # define 3 layers: input, hidden, and output, all Linear transformations
        self.first_linear = nn.Linear(input_size, hidden_lyr_size)
        self.second_linear = nn.Linear(hidden_lyr_size, output_size)

    def forward(self):
        # simple forward propagation: Relu activation function after linearly transforming input layer, then a further L.T. to the output layer.
        x = F.relu(self.first_linear)
        x = self.second_linear(x)

        return x

    def save(self, file_name="model.pth"):
        folder_path = "./model"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)

        # Captures a state of the NN, including weights/biases etc.
        torch.save(self.state_dict(), file_name)

class Trainer():
    def __init__(self, learning_rate, disc_rate, model):
        self.lr = learning_rate
        self.disc_rate = disc_rate
        self.model = model
        # Define optimiziation algorithm: makes it far easier to adjust hyperparamters to reduce loss.

        

