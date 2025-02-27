import torch 
import torch.nn as nn
import torch.nn.functional as F

# import pdb   # debugger

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
        hidden_layers: list of torch.nn modules ()
        # TODO: input_layer, output_layer:
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
        self.input_layer = nn.Linear(self.N_in, self.N(1), dtype=torch.float64)
        self.hidden_layers = nn.ModuleList()

        for i in range(self.N_layers-2):
            self.hidden_layers.append(nn.Linear(self.N(i+2), self.N(i+2), dtype=torch.float64)) 

        self.output_layer = nn.Linear(self.N(self.N_layers), self.N_out, dtype=torch.float64)


    def forward(self,x):
        # ReLU on input layer
        x = F.relu(self.input_layer(x))

        # Skip ReLU on the output layer
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))

        x = self.output_layer(x)
        # pass the logits from penultimate hidden layer to output layer
        # x = self.hidden_layers[-1](x)
        return x
    
    def fit(self,train_data, optimizer,loss_criterion,epochs=5):
        '''
            Train the neural network on the input 'train_data'. 
            Adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network (Accessed 24/02/2025)
        '''
        self.train()
        train_acc = 0
        total = 0
        for epoch in range(epochs):  

            running_loss = 0.0
            for batch_idx, (data,target) in enumerate(train_data):

                # breakpoint()
                
                optimizer.zero_grad()

                outputs = self.forward(data)
                loss = loss_criterion(outputs, target)
                loss.backward()
                optimizer.step()

                _, preds  = torch.max(outputs, dim=1)


                train_acc += torch.sum(preds == target)
                total += len(preds)

                # print statistics
                running_loss += loss.item()
                if batch_idx % 100 == 0:    
                    print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f} --- acc: {train_acc / total}')
                    running_loss = 0.0

        print('Finished Training')


def torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU network '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)

