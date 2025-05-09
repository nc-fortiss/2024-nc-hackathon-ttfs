import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import config_utils
import h5py
import pickle
from train_torch import evaluate_FC_SNN
import matplotlib.pyplot as plt

# import pdb   # debugger

'''
    Module implementing the pytorch version of the neural network architectures. 
'''

def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    # if config_utils.DEBUG_MODE:
    #     breakpoint()
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    # Calculate output spiking time ti (Eq. 7)
    ### Debugging only ###
    
    # print(f"call spiking --- t_min={t_min} --- t_max={t_max} --- thresh={threshold[0]} --- tj.shape={tj.shape}")
    # breakpoint()

    ti = (torch.matmul(tj-t_min, W) + threshold + t_min)

    # Ensure valid spiking time. Do not spike for ti >= t_max.
    # No spike is modelled as t_max that cancels out in the next layer (tj-t_min) as t_min there is t_max
    # if config_utils.DEBUG_MODE: breakpoint()
    # ti = torch.where(ti < t_max, ti, t_max)
    # TODO: how to set ti if ti >= t_max_quantized
    ti = torch.where(ti < robustness_params['latency_quantiles'] * t_max, ti, t_max)
    # Add noise to the spiking time for noise simulations
    ti = ti + torch.normal(mean=0.0, std=robustness_params['noise'], size=ti.shape, dtype=torch.float64)
    return ti

def compute_membrane_potential(t, i, tj, W):
    '''
        Evaluates the membrane potential at the time point (t) for neuron(i) in layer N
        given input spike times (tj), the boundary (t_min) and the kernel (W) from layer N.
        Only spike times up to (t) are considered for the evaluation of the membrane potential,
        given that the input spike time array (tj) already contains all the spikes from layer (N-1)
    
    '''
    # breakpoint()
    W_i = W[:, i]         # filter for all synapses connected to neuron (i) in Layer N 
    
    mask = (tj <= t)     # only spikes earlier than (t) have already contributed to V

    previous_spikes = np.where(mask, tj, 0.0)    # only active spikes keep value, else 0.0
    previous_active_weights = np.where(mask, W_i, 0.0)    # only active weights

    shift = (t - previous_spikes)   # shift relative distance between current (t) and (tj)
    weighted_sum = np.matmul(shift, previous_active_weights)        # W * (t-tj) for active pre-neurons j

    return weighted_sum 


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
        
        # Trainable threshold parameter
        self.D_i = nn.Parameter(torch.zeros(N_out))

        # Vectors with slope and membrane potential for each neuron for discretized network
        self.A = None
        self.V = None



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
        
        return ti
    
    def discrete_forward(self, delta_k, input_spikes):
        '''
            First, the 'input_spikes' arriving from the previous layer are integrated 
            in the membrane potential of the neurons in this layer 
            in a step-wise manner, with step size equal to 'delta_k'. Next, the membrane 
            potentials follow the constant update rule. If a neuron's membrane potential 
            reaches the threshold at a discrete timestep K, then a '1' will be placed in the 
            respective index of the output spikes tensor. 

            Arguments:
            delta_k: defines the step size; it should be equal across all layers
            input_spikes: shape=[K, B, N] for timestep index K in prev layer, batch B and neuron N in current layer, 
                        containing a '1' for every neuron index that produced a spike at time K in the previous layer. 
        '''

        # Number of discrete timesteps that can happen before t_max in the spiking window interval of this (!) layer
        number_timesteps = int((self.t_max - self.t_min) / delta_k) 
        batch_size = input_spikes.shape[1]

        discrete_output_tensor = torch.zeros([number_timesteps, batch_size, self.N_out], dtype=torch.float64)
        layer_threshold = self.t_max - self.t_min 

        # 1. Process the in-coming spikes from the previous layer: integration phase 
        # Iterate over each timestep from the previous layer
        for timestep_index in range(input_spikes.shape[0]):
            batched_spikes_matrix = input_spikes[timestep_index]

            weighted_spikes = torch.matmul(batched_spikes_matrix, self.kernel)
            self.A = self.A + weighted_spikes
            self.V = self.V + self.A * delta_k
        
        # 2. If this is not the output layer: switch to spiking phase, starting at layer.t_min
        # Fix constant slope and populate the output spikes vector as needed
        if not self.is_output:
            timestep = self.t_min
            for timestep_index in range(number_timesteps):
                self.V = self.V + delta_k 

                # Check spike condition: get indices of neurons that passed threshold 
                spiking_neurons_mask = self.V >= layer_threshold

                # Update output tensor at this timestep
                discrete_output_tensor[timestep_index] = spiking_neurons_mask.float()

                # Reset membrane potential of neurons that have spiked (neurons fire at most once)
                self.V[spiking_neurons_mask] = 0.0
                timestep += delta_k
            return discrete_output_tensor
        else:
            # at the (non-spiking) output layer, we return the membrane potentials, which can be used for prediction
            return self.V


class FC_ReLU_torch(nn.Module):
    ''' Defines instance of a fully-connected ReLU network

    Attributes:
        N_layers: number of hidden layers (excluding input and output layers)
        N_hid: number of neurons in hidden layer(s); can be of type (int) or List[int]
        N_in, N_out: number of neurons at the input / output layers
        N: returns N_hid[l] if N_hid is a list, else returns the constant N_hid value
        hidden_layers: list of torch.nn modules ()
        relu: torch definition of the relu activation function
        max_activations: tracks the global maximzum output activation values for each layer in a list
    ''' 
    def __init__(self, layers, N_hid,N_in, N_out):
        super().__init__()
        self.N_layers=layers 
        self.N_hid = N_hid
        self.N_in = N_in 
        self.N_out = N_out 
        self.N = lambda l: (N_hid[l-1] if type(N_hid)==list else N_hid)
        self.collect_activations = False 
        

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

        # Add relu definition
        self.relu = nn.ReLU()

        # Register forward hooks on each hidden layer
        self.max_activations = {}
        self.all_activations = {}
        for i, layer in enumerate(self.hidden_layers[:-1]):
           layer_name = f"layer_{i}"
           layer.register_forward_hook(self.get_max_activation(layer_name))
           layer.register_forward_hook(self.get_activations(layer_name))

    def forward(self,x):
        # Skip ReLU on the output layer

        for layer in self.hidden_layers[:-1]:
            x = self.relu(layer(x))

        # pass the logits from penultimate hidden layer to output layer
        x = self.hidden_layers[self.N_layers-1](x)
        return x
    

    def get_max_activation(self, name):
        '''
            Returns the maximum activation for a single layer during a forward pass.
            The max is updated for each individual batch (at each forward call). 
            Therefore, when predicting with a dataset, model.max_activations will store
            the maximum activation value across all batches in the training set.

            Function implementation also adapted from: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#using-the-forward-hooks (Accessed 27/03/25)

            @name: string, name of the layer
        '''
        def hook(model, input, output):
            relu_output = F.relu(output)
            batch_max = torch.max(relu_output).item()
            if name not in self.max_activations:
                self.max_activations[name] = batch_max
            else:
                self.max_activations[name] = max(self.max_activations[name], batch_max)

        return hook
    
    def get_activations(self, layer_name):
        '''
            Collects the activations from the outputs of each layer. 
            @layer_name: string, name of the layer
        '''
        def hook(model, input, output):
            # Collect the output spikes time during a forward call
            if self.collect_activations:
                if layer_name not in self.all_activations:
                    self.all_activations[layer_name] = []     
                self.all_activations[layer_name].extend(output.flatten().detach().cpu().tolist())
        return hook


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
        output_layer: the (non-spiking) output layer; pass X_n := 1
        min_spike_times: collects the minimum spiking timestamp per individual layer for a single
            forward pass; it is reset after each forward() call and is used to update t_max in training
        
    
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

        # Add output layer separetely
        self.output_layer = SpikingDenseTorch(self.N(self.N_layers), self.N_out, robustness_params=robustness_params, is_output=True)

        # Register forward hooks on each hidden layer to capture intermediate output activations layer-wise
        self.activations = {}
        self.min_spike_times = {}
        self.collect_activations = False    # set in the main to control when outputs are collected
        for i, layer in enumerate(self.hidden_layers):
           layer_name = f"layer_{i}"
           layer.register_forward_hook(self.get_min_spiketime(layer_name))
           layer.register_forward_hook(self.get_activations(layer_name))
        
        self.discretization = False 
        self.delta_k = None 
           
    def forward(self, x):
        ''' 
            Defines the forward pass through the entire SNN architecture 
        '''

        if not self.discretization:
            for i, l in enumerate(self.hidden_layers):
                x = l(x)
            x = self.output_layer(x)
            return x 
        else: 
            discrete_x = self.discrete_forward(self.delta_k, x)
            return discrete_x
 
    def set_snn_intervals(self, t_min_start=0, t_max_start=1):
        ''' Helper function to create the [t_min, t_max] boundaries for the 
            integrate vs spike time windows for each layer. 
            't_min_start' and 't_max_start' define the min/max time values in the input layer. 
        '''
        t_min, t_max= t_min_start, t_max_start
        layer_num = 0
        for child in self.children():
            if isinstance(child, nn.ModuleList):    # the hidden layers appear under a single child node as a moduleList
                for layer in child: 
                    t_min, t_max = layer.set_intervals(t_min, t_max)
                    layer_num += 1
            else: 
                t_min, t_max = child.set_intervals(t_min,t_max)    # for the output layer 
                layer_num+=1 

    def get_min_spiketime(self, layer_name):
        '''
            Returns the minimum (first) spiking timestamp for a single layer during a forward pass.
            This minimum is updated for each new forward call. It is then used for updating t_max
            when a forward call is made on each new batch while training the SNN.
         
            @layer_name: string, name of the layer
        '''
        def hook(model, input, output):
            # Collect the minimum layer output spike time for this batch on this forward call
            forward_output_min = torch.min(output).item()
            self.min_spike_times[layer_name] = forward_output_min

        return hook

    def get_activations(self, layer_name):
        '''
            Collects the activations from the outputs of each layer. In this case
            the outputs are the spike time-stamps. Activations are collected only when 
            'collect_activations' is set in the main, to control the collection process. 

            @layer_name: string, name of the layer
        '''
        def hook(model, input, output):
            # Collect the output spikes time during a forward call
            if self.collect_activations:
                if layer_name not in self.activations:
                    self.activations[layer_name] = []     
                self.activations[layer_name].extend(output.flatten().detach().cpu().tolist())

        return hook

    def dump_activations(self, path):
        '''
        Dumps all the layer-wise collected activations into the .npy file specified
        at 'path'. It also clears the dictionary 'self.activations' to make space for 
        later further collections. 
        '''
        with open(path, "wb") as f:
            pickle.dump(self.activations, f)
        self.activations = {}
            
        # TODO: to ease saving/loading, construct the absolute constant path inside function
        # and pass the file name only as input. Same should change in plotting function

    def apply_max_quantiles(self, quantile):
        '''
            Apply latency quantiles to the t_max boundary of each layer. This simulates shifting t_max
            closer to t_min without actually modifying the boundary. If an input spike arrives at or 
            right after (t_max*quantile), this spike is considered as irrelevant. 
            # TODO: explain this better - what is the advantage of doing this?
        '''
        quantile = quantile/100
        for layer in self.hidden_layers:
            layer.robustness_params["latency_quantiles"] = quantile
        
    def optimize_threshold(self, test_data):
        '''
            The model is evaluated on the test data; from this pass, the global minimum spike time 
            across all samples is collected for each layer, which is used as a reference to shift the 
            threshold and the intervals so that, with the shifted intervals, the first spike occurs
            exactly at t_min (for the sample that produced the same global minimum)

        '''
        extend_margin = 0.1
        self.min_spike_times = {}       # reset minimum spike times
        self.collect_activations = False        # no need to collect activations during evaluation here
        evaluate_FC_SNN(self, test_data)
        config_utils.logging.info(f"Global minimum spike times for test_data: {self.min_spike_times}")

        t_max_new = 1
        for i, layer in enumerate(self.hidden_layers):
            layer_name = f'layer_{i}'
            layer.t_min=t_max_new
            t_max_new = layer.t_max + layer.t_min - self.min_spike_times[layer_name]
            t_max_new = (1 + extend_margin) * t_max_new
            layer.t_max=t_max_new

        self.output_layer.t_min = t_max_new             # the output layer interval has a size of 1.5 by default
        self.output_layer.t_max = t_max_new + 1.5

    def discrete_forward(self, delta_k, input_spikes):
        
        # Setup layer-wise parameters based on input shape
        batch_size = input_spikes.shape[0]
        for layer in self.hidden_layers:
            layer.A = torch.zeros(batch_size, layer.N_out, device=input_spikes.device, dtype=torch.float64)
            layer.V = torch.zeros(batch_size, layer.N_out, device=input_spikes.device, dtype=torch.float64)

        # Setup parameters for output layer
        self.output_layer.A = torch.zeros(batch_size, self.output_layer.N_out, device=input_spikes.device, dtype=torch.float64)
        self.output_layer.V = torch.zeros(batch_size, self.output_layer.N_out, device=input_spikes.device, dtype=torch.float64)

        # breakpoint()

        # 1. Discretize the input spikes that have been encoded
        number_steps_input_layer = int(1/delta_k)
        discrete_3d_spike_tensor = torch.zeros((number_steps_input_layer, *input_spikes.shape), dtype=torch.float64)

        timestep_index = 0
        for timestep_index in range(number_steps_input_layer):
            lower_bin = timestep_index * delta_k                # define bin boundaries
            upper_bin = (timestep_index + 1) * delta_k
            mask = (input_spikes >= lower_bin) & (input_spikes < upper_bin) # filter tensor
            discrete_3d_spike_tensor[timestep_index] = mask.float()            # add to discretized tensor

        # 2. Save discretized matrices for each batch for debugging
        # !!! For PLOTTING only
        # for batch_index in range(discrete_3d_spike_tensor.shape[1]):
        #     spikes_first_batch = discrete_3d_spike_tensor[:, batch_index, :]
        #     save_path = config_utils.LOGGING_DIR + f'/outputs/discrete_input_batch_{batch_index}.txt'
        #     np.savetxt(save_path, spikes_first_batch, fmt="%d")


        # 3. Send the discretized input to 1st hidden layer and then iterate over the remaining layers
        batch_index = 0
        for i,layer in enumerate(self.hidden_layers):
            discrete_3d_spike_tensor = layer.discrete_forward(delta_k, discrete_3d_spike_tensor)

            # !!! For PLOTTING only - save a sample from the 1st batch for visualization
            # batch_spike_tensor = discrete_3d_spike_tensor[:, batch_index, :]
            # save_path = config_utils.LOGGING_DIR + f'/outputs/discrete_layer_{i}_batch_{batch_index}.txt'
            # np.savetxt(save_path, batch_spike_tensor, fmt="%d")

        # Pass tensor from last hidden layer to output layer, which returns the read-out membrane potential values
        final_output_tensor = self.output_layer.discrete_forward(delta_k, discrete_3d_spike_tensor)
        return final_output_tensor
   

def create_torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU model '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)

def create_torch_fc_model_SNN(layers=2, N_hid=340, N_in=784, N_out=10, X_n=1000, robustness_params={}):
    ''' Returns an instance of a fully-connected SNN model '''
    return FC_SNN_torch(layers,N_hid,N_in,N_out,X_n,robustness_params=robustness_params, kernel_regularizer=None, kernel_initializer=None)
