from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        # Updates the weights in the NN based on training data.
        self.optimiser = optim.Adam(model.parameters(), lr=self.lr)
        # Mean squared error used to compute loss
        self.loss = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, done):
        # initialize all the parameters as tensors
        self.state = torch.tensor(state, dtype=torch.float)
        self.next_state = torch.tensor(next_state, dtype=torch.float)
        self.action = torch.tensor(action, dtype=torch.long)
        self.reward = torch.tensor(reward, dtype=torch.float)
        


        # Check size of training data (as it can be single or in a batch)
        # Shape of tensor should be in form (n, x), n = #number of batches, x = # amount of data in each batch
        # If a single set is used, len of the tensor shape will be 1, so check this.
        if len(self.state.shape) == 1:
            # Need to add a (empty) dimension to each tensor to fit into correct shape as batches are expected.
            self.state = torch.unsqueeze(self.state, 0)
            self.next_state = torch.unsqueeze(self.next_state, 0)
            self.action = torch.unsqueeze(self.action, 0)
            self.reward = torch.unsqueeze(self.reward, 0)
            done = (done, )

        # Make prediction based on the model
        prediction = self.model(state)

        
    
        

