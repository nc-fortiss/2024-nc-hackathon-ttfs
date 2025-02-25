import torch 
import torch.nn as nn
import torch.nn.functional as F

'''
    Module implementing the pytorch version of the neural network architectures. 
'''


class SpikingDenseTorch(nn.Module):
    ''' Creates a single Spiking Dense Layer '''

    def __init__(self):
        super().__init__()
    
    # TODO: define Spiking Layers in torch


class FC_ReLU_torch(nn.Module):
    ''' Defines instance of a fully-connected ReLU network

    Attributes:
        N_layers: number of hidden layers (excluding input and output layers)
        N_hid: number of neurons in hidden layer(s); can be of type (int) or List[int]
        N_in, N_out: number of neurons at the input / output layers
        N: returns N_hid[l] if N_hid is a list, else returns the constant N_hid value
        layers_list: list of torch.nn modules ()
    '''
    def __init__(self, layers, N_hid,N_in, N_out):
        super().__init__()
        self.N_layers=layers 
        self.N_hid = N_hid
        self.N_in = N_in 
        self.N_out = N_out 
        self.N = lambda l: (N_hid[l-1] if type(N_hid)==list else N_hid)

        # Dynamically adjust the configuration of hidden layers based on input.
        # The definition of an Input() dummy tensor as done in the tensorflow implementation can be skipped; here the first hidden layer is created directly
        # Adapted from: https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124 (accessed 24/02/2025)
        self.layers_list = nn.ModuleList()

        l = 1
        self.layers_list.append(nn.Linear(self.N_in, self.N(l), dtype=torch.float64))       # 1st hidden layer
        l += 1

        while l < (self.N_layers-2):
            self.layers_list.append(nn.Linear(self.N(l+2), self.N(l+2), dtype=torch.float64))      # further hidden layers

        self.layers_list.append(nn.Linear(self.N(l), self.N_out, dtype=torch.float64))      # output layer

    def forward(self,x):
        for layer in self.layers_list:
            x = F.relu(layer(x))

        return x


def torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU network '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)
