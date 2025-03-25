import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import config_utils
import h5py

# import pdb   # debugger

'''
    Module implementing the pytorch version of the neural network architectures. 
'''

def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    # Calculate output spiking time ti (Eq. 7)
    ### Debugging only ###
    '''
    print("call spiking")
    print("t_min=", t_min)
    print("t_max=", t_max)
    print("tj=", tj)


    print("D_i=",D_i)
    print("thres=", threshold)
    print("\n")
    # breakpoint()
    '''
    ti = (torch.matmul(tj-t_min, W) + threshold + t_min)

    # Ensure valid spiking time. Do not spike for ti >= t_max.
    # No spike is modelled as t_max that cancels out in the next layer (tj-t_min) as t_min there is t_max
    ti = torch.where(ti < t_max, ti, t_max)
    # Add noise to the spiking time for noise simulations
    ti = ti + torch.normal(mean=0.0, std=robustness_params['noise'], size=ti.shape, dtype=torch.float64)
    return ti

class SpikingDenseTorch(nn.Module):
    ''' Creates a single Spiking Dense Layer 
    
        Attributes:
            N_in, N_out: number of input/output neurons of this layer
            X_n: together with 'B_n', regulates spiking window size for this layer. 
                In case of the output layer, then 'B_n' is kept small to avoid spike-generation as these spikes are not relevant anymore
            t_min_prev: corresponds to the 't_min' from the spiking-time-window of the previous layer
            t_min, t_max: represent the minimum and maximum spike times for the spiking-time-window of the current layer
            robustness_params = # TODO
            alpha: # TODO
            regularizer = # TODO
            initializer = # TODO
            kernel: # TODO // trainable neuron weight
            is_output: boolean flag, True if self is the output layer
            D_i: # TODO 

    '''

    def __init__(self, N_in, N_out, X_n=1, robustness_params={}, kernel_regularizer=None, kernel_initializer=None, is_output=False):
        super().__init__()

        self.N_in = N_in            
        self.N_out = N_out
        self.X_n = X_n
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.is_output = is_output

        self.robustness_params=robustness_params
        self.alpha = torch.ones(self.N_in, dtype=torch.float64)
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer

        self.kernel = nn.Parameter(torch.empty((N_in,N_out), dtype=torch.float64))    # register weight in model
        if kernel_initializer:      # TODO: allow custom initializer
            i = 0
        else: 
            init.xavier_uniform_(self.kernel)           # initialize weight tensor, with initializer if provided

        self.D_i = nn.Parameter(torch.zeros(N_out))
        
    def set_intervals(self, t_min_prev,t_min):
        ''' Sets t_min_prev, t_min, and t_max for this layer. The bounds are determined and set 
            before training even begins. Equivalent to 'set_params' in the tensorflow code. 
        
        '''
        self.t_min_prev = t_min_prev 
        self.t_min = t_min 
        self.t_max = t_min + self.B_n 
        return t_min, t_min+self.B_n

    
    def forward(self, tj):
        ''' Defines a single pass through the Spiking Dense Layer.
            The input 'tj' represents the spike times that were integrated 
            from the previous layer, thus transforming them into output spike times 'ti'
        '''
        ti = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        if (self.is_output):
            W_mult_x = torch.matmul(self.t_min-tj, self.kernel)
            self.alpha = self.D_i/(self.t_min-self.t_min_prev)
            ti = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x
        
        # save TI times per layer in the logs if debugging is enabled 
        # skip output layer since no more spikes will be produced
        if config_utils.DEBUG_MODE and not self.is_output:
            # print("-----------------------------------------------------------------")
            # config_utils.logging.info("Writing TI's outputs after call_spiking")
            # config_utils.logging.info(ti.tolist())

            filename = config_utils.LOGGING_DIR + 'spike_output.txt'
            with open (filename, 'a+') as f:
                f.write(' '.join(str(ti) for ti in ti.tolist()))        # convert tensor to list and separate values by a ' '
                f.write('\n')

        return ti



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
        # self.output_layer = nn.Linear(self.N(self.N_layers), self.N_out, dtype=torch.float64)
        # TODO: choose if to keep the output layer separate from the list (might be possible to be intetgrated)
        self.hidden_layers.append(nn.Linear(self.N(layers), self.N_out, dtype=torch.float64))

        self.relu = nn.ReLU()

    def forward(self,x):
        # Skip ReLU on the output layer

        for layer in self.hidden_layers[:-1]:
            x = self.relu(layer(x))

        # pass the logits from penultimate hidden layer to output layer
        # x = self.output_layer(x)
        x = self.hidden_layers[self.N_layers-1](x)
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

    # TODO: define min_ti's as in tensorflow    
    
    '''
    def __init__(self, layers, N_hid, N_in, N_out, X_n, robustness_params, kernel_regularizer, kernel_initializer):
        super().__init__()
        self.N_layers=layers 
        self.N_hid = N_hid
        self.N_in = N_in 
        self.N_out = N_out 
        self.X_n = X_n

        self.N = lambda l: (N_hid[l-1] if type(N_hid)==list else N_hid)

        # Initialize list of hidden layer modules and append 1st default hidden layer
        self.hidden_layers = nn.ModuleList() 
        self.hidden_layers.append(SpikingDenseTorch(self.N_in, self.N(1), (X_n[0] if type(X_n)==list else X_n), robustness_params=robustness_params))

        for i in range(self.N_layers-2):        # If N_layers > 2, append the rest of the layers
            self.hidden_layers.append(SpikingDenseTorch(self.N(i+2), self.N(i+2), (X_n[i+1] if type(X_n)==list else X_n), robustness_params=robustness_params)) 


        self.output_layer = SpikingDenseTorch(self.N(self.N_layers), self.N_out, robustness_params=robustness_params, is_output=True)

        # Store the layer-wise spike-time outputs as a list of lists
        self.layer_activations = [ np.empty([0]) for _ in range(len(self.hidden_layers)) ]
        # self.list_activations = nn.Parameter
        self.list_activations = [ [] for _ in range(len(self.hidden_layers))]

        # Register hooks to capture activations
        # self.activations = {f"layer_{i}": [] for i in range(len(self.hidden_layers))}
        # for i, layer in enumerate(self.hidden_layers):
          #  layer.register_forward_hook(self._create_hook_fn(f"layer_{i}"))

    def forward(self, x):
        ''' Defines the forward pass through the entire SNN architecture '''

        # Store the activations for this current batch
        # activations_dict = {} 
        for i, l in enumerate(self.hidden_layers):
            x = l(x)

            # breakpoint()
            ''' use: self.layer_activations[i] = np.append(self.layer_activations[i], x.flatten().detach().numpy()) '''
            # self.list_activations[i].extend(x.flatten().detach().tolist())

            if config_utils.DEBUG_MODE:
                self.layer_activations[i] = np.append(self.layer_activations[i], x.flatten().detach().numpy())
                self.list_activations[i].extend(x.flatten().detach().tolist())
                # activations_dict[f"layer_{i}"] = x.detach().cpu().numpy() 

        # save_path = config_utils.LOGGING_DIR + 'activations.npz'
        # np.savez(save_path, **activations_dict)
        x = self.output_layer(x)
        return x 
    
    def set_snn_intervals(self, t_min_start=0, t_max_start=1):
        ''' Helper function to create the [t_min, t_max] boundaries for the 
            integrate vs spike time windows for each layer. 
            't_min_start' and 't_max_start' define the min/max time values in the input layer. 
        '''
        t_min, t_max= t_min_start, t_max_start
        layer_num = 0
        config_utils.logging.info("\n")
        for child in self.children():
            if isinstance(child, nn.ModuleList):    # the hidden layers appear under a single child node as a moduleList
                for layer in child: 
                    t_min, t_max = layer.set_intervals(t_min, t_max)
                    config_utils.logging.info(f"    hidden layer_num={layer_num} -> B_n = {layer.B_n:>7.2f}; t_min_prev={layer.t_min_prev:>7.2f};   t_min={layer.t_min:>7.2f}; t_max={layer.t_max:>7.2f}")
                    layer_num += 1
            else: 
                t_min, t_max = child.set_intervals(t_min,t_max)    # for the output layer 
                config_utils.logging.info(f"    output layer_num={layer_num} -> B_n = {child.B_n:>7.2f}; t_min_prev={child.t_min_prev:>7.2f};   t_min={child.t_min:>7.2f}; t_max={child.t_max:>7.2f}\n")
                layer_num+=1 

    def _create_hook_fn(self, layer_name):
        ''' Creates a hook function for a specific layer '''
        def hook_fn(module, input, output):
            self.activations[layer_name].append(output.detach().cpu().numpy())
        return hook_fn
    
    def save_activations(self, save_path):
        ''' Saves activations to an HDF5 file '''
        with h5py.File(save_path, 'w') as f:
            for layer_name, activation_list in self.activations.items():
                # Concatenate all activations for this layer
                activations = np.concatenate(activation_list, axis=0)
                f.create_dataset(layer_name, data=activations, dtype='float32')
        

def create_torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU model '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)

def create_torch_fc_model_SNN(layers=2, N_hid=340, N_in=784, N_out=10, X_n=1000, robustness_params={}):
    ''' Returns an instance of a fully-connected SNN model '''
    return FC_SNN_torch(layers,N_hid,N_in,N_out,X_n,robustness_params=robustness_params, kernel_regularizer=None, kernel_initializer=None)
