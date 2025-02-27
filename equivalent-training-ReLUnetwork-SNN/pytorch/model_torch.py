import torch 
import torch.nn as nn
import torch.nn.functional as F

# import pdb   # debugger

'''
    Module implementing the pytorch version of the neural network architectures. 
'''


class SpikingDenseTorch(nn.Module):
    ''' Creates a single Spiking Dense Layer 
    
        Attributes:
            N_in, N_out: number of input/output neurons of this layer
            X_n: together with 'B_n', regulates spiking window size for this layer. 
                In case of the output layer, then 'B_n' is kept small to avoid spike-generation as these spikes are not relevant anymore
            t_min_prev: 't_min' from the spiking-time-window of the previous layer
            t_min, t_max: 't_min' and 't_max' timestamps of the current layer
            robustness_params = # TODO
            alpha: # TODO
            regularizer = # TODO
            initializer = # TODO
    '''

    def __init__(self, N_in, N_out, X_n=1, robustness_params={}, kernel_regularizer=None, kernel_initializer=None):
        super().__init__()

        self.N_in = N_in
        self.N_out = N_out
        self.X_n = X_n
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.robustness_params=robustness_params
        self.alpha = torch.ones(self.N_in, dtype=torch.float64)
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer

    def set_intervals(self, t_min_prev,t_min):
        ''' Sets t_min_prev, t_min, and t_max for this layer. The bounds are determined and set 
            before training even begins. Equivalent to 'set_params' in the tensorflow code. 
        
        '''
        self.t_min_prev = t_min_prev 
        self.t_min = t_min 
        self.t_max = t_min + self.B_n 
        return t_min, t_min+self.B_n

    






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
        self.hidden_layers = nn.ModuleList()

        # Add the 1st hidden layer by default
        self.hidden_layers.append(nn.Linear(self.N_in, self.N(1), dtype=torch.float64))

        # If layers > 2 (default), keep adding fully-connected Dense layers 
        for i in range(self.N_layers-2):
            self.hidden_layers.append(nn.Linear(self.N(i+2), self.N(i+2), dtype=torch.float64)) 

        # Add output layer separately
        self.output_layer = nn.Linear(self.N(self.N_layers), self.N_out, dtype=torch.float64)


    def forward(self,x):
        # Skip ReLU on the output layer
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))

        # pass the logits from penultimate hidden layer to output layer
        x = self.output_layer(x)
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

class FC_SNN_torch(nn.Module):
    ''' Defines instance of a fully-connected SNN model

    Attributes:
        N_layers: number of hidden layers
        N_hid: number of neurons in hidden layers. Can be of type (int) or List[int]
        N_in: number of neurons in the input layer
        N_out: number of neurons in the output layer (number of classes)
        X_n: window-scaling factor (TODO: clarify)
        N(l): lambda to extract 'N_hid' if it is a list
        hidden_layers: list of nn.Module hidden layers

    
    
    '''
    def __init__(self, layers, N_hid, N_in, N_out, X_n, robustness_params, kernel_regularizer, kernel_initializer):#
        super().__init__()
        self.N_layers=layers 
        self.N_hid = N_hid
        self.N_in = N_in 
        self.N_out = N_out 
        self.X_n = X_n

        self.N = lambda l: (N_hid[l-1] if type(N_hid)==list else N_hid)

        # Initialize list of hidden layer modules and append 1st default hidden layer
        self.hidden_layers = nn.ModuleList() 
        self.hidden_layers.append(SpikingDenseTorch(self.N_in, self.N(1), (X_n[0] if type(X_n)==list else X_n)))

        for i in range(self.N_layers-2):
            self.hidden_layers.append(SpikingDenseTorch(self.N(i+2), self.N(i+2), (X_n[i+1] if type(X_n)==list else X_n))) 


        self.output_layer = SpikingDenseTorch(self.N(self.N_layers), self.N_out, robustness_params=robustness_params)

        

def create_torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU model '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)

def create_torch_fc_model_SNN(layers=2, N_hid=340, N_in=784, N_out=10, X_n=1000, robustness_params={}):
    ''' Returns an instance of a fully-connected SNN model '''
    return FC_SNN_torch(layers,N_hid,N_in,N_out,X_n,robustness_params=robustness_params, kernel_regularizer=None, kernel_initializer=None)
