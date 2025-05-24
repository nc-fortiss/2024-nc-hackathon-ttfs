import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import config_utils
import h5py
import math
import pickle
import matplotlib.pyplot as plt

# import pdb   # debugger

'''
    Module implementing the pytorch version of the neural network architectures. 
'''

def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    # if config_utils.DEBUG_MODE:
    #     breakpoint()
    # Calculate the spiking threshold (Eq. 18)
    # breakpoint()
    threshold = t_max - t_min - D_i
    # Calculate output spiking time ti (Eq. 7)
    ### Debugging only ###
    
    # print(f"call spiking --- t_min={t_min} --- t_max={t_max} --- thresh={threshold[0]} --- tj.shape={tj.shape}")
    # breakpoint()
    
    # print(f"tj.shape={tj.shape} --- W.shape={W.shape}")
    ti = (torch.matmul(tj-t_min, W) + threshold + t_min)

    # Ensure valid spiking time. Do not spike for ti >= t_max.
    # No spike is modelled as t_max that cancels out in the next layer (tj-t_min) as t_min there is t_max
    # if config_utils.DEBUG_MODE: breakpoint()
    # ti = torch.where(ti < t_max, ti, t_max)
    # TODO: how to set ti if ti >= t_max_quantized
    
    ti = torch.where(ti < t_max, ti, t_max)
    print(f"spikes={(ti < t_max).sum().item()}")

    # Add noise to the spiking time for noise simulations
    ti = ti + torch.normal(mean=0.0, std=0.0, size=ti.shape, dtype=torch.float64)
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

def extract_patches_from_spike_tensor(tj, ):

        ''' Apply PATCHING to input spike tensor 'tj' with shape [B, H, W, C] 

            Extracts image patches of size (kH, kW) across all input channels C which are S stride(s) apart from each other. 
            This simulates the convolutional sliding window. Each patch is flattened into a 1D vector of shape P=(kH * kW * C, 0)
            The operation then collects each patch vector into the re-shaped output tensor of shape [B, O, O, P]
            where O stands for the size of each output activation filter.   

            For example, for a padded CIFAR10 input spike tensor with shape [B, 34, 34, 3] and a kernel size (3,3):
            As one patch also extracts the features from the channels, each patch will have a (flattened) shape of O = ((3*3) * 3) = (27) 
            We can extract 32 patches in the width direction and 32 in the height direction. 
            In total, all patches are collected in the output tensor with shape [B, 32, 32, 27]

            In order to subset the input tensor, the torch.Tensor.unfold function can be used. 

        ''' 
        return 0 

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

    def __init__(self, N_in, N_out, X_n=1, name="", robustness_params={}, kernel_regularizer=None, kernel_initializer=None, is_output=False):
        super().__init__()

        self.N_in = N_in            
        self.N_out = N_out
        self.X_n = X_n
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.is_output = is_output
        self.name=name

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
        # breakpoint()
        # config_utils.DEBUG_MODE = True
        ti = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        if (self.is_output):
            # breakpoint()
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

class SpikingConv2DTorch(nn.Module):
    def __init__(self, filters, in_channels, robustness_params, name=None, X_n=1, padding='same', kernel_size=(3,3),
                 kernel_regularizer=None, kernel_initializer=None):

        super().__init__()
        self.in_channels = in_channels
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max=0.0, 0.0, 1.0
        self.robustness_params=robustness_params
        self.alpha = torch.ones(filters, dtype=torch.float64)
        self.name=name 

        self.first_convolutional_layer = False 
        self.padding_per_side = kernel_size[0] // 2 if padding == 'same' else 0     

        # Filter weight
        init_kernel = torch.empty((*self.kernel_size, in_channels, filters), dtype=torch.float64)
        self.kernel = nn.Parameter(init_kernel, requires_grad=True)  

        if self.initializer:
            self.initializer(self.kernel)
        else:
            init.xavier_uniform_(self.kernel)

        # BN fusion flags (non-trainable)
        self.BN = 0 
        self.BN_before_ReLU = 0

        # D_i: shape = (9, filters) for different padding cases
        self.bias = nn.Parameter(torch.zeros((9, filters), dtype=torch.float64), requires_grad=True)

    def set_intervals(self, t_min_prev,t_min):
        ''' Sets t_min_prev, t_min, and t_max for this layer. The bounds are determined and set 
            before training even begins. Equivalent to 'set_params' in the tensorflow code. 
        '''
        self.t_min_prev = t_min_prev 
        self.t_min = t_min 
        self.t_max = t_min + self.B_n 
        return t_min, t_min+self.B_n

    def forward(self, tj):
        """
        Input spiking times tj: [B, H, W, C]
        Output spiking times ti. 
        """
        # print(f"### layer.name={self.name} - input.shape={tj.shape}")


        image_original_size = tj.shape[2]       # image size with no padding (as input) 
        image_valid_size = image_original_size - self.kernel_size[0] + 1    # output filter size if 'valid' padding is used (no change)
        image_same_size = tj.shape[2]

        kH = self.kernel_size[0]        # kernel height
        kW = self.kernel_size[1]        # kernel width

        # If the input spike times are the result from the previous layer, then the shape is [B, H, W, F]
        # where (H,W) is the activation filter size and F is the number of filters. In this case the tensor's
        # shape needs to be adapted for the F.pad() interface. The spike times at the input are already correct.
        if not self.first_convolutional_layer:    
            tj.permute(0,3,1,2)         # [B, H, W, F] -> [B, F, H, W]


        # Add spatial padding to the spike times: pad=(left, right, top, bottom) 
        # pad with 't_min' as it's equivalent to the relative 0 spike time in this layer
        tj = F.pad(tj, pad=(1, 1, 1, 1), mode='constant', value=self.t_min)

        tensor_batch = tj.clone().detach().squeeze(0)  # remove batch dimension → [C, H, W]

    
        ''' PATCHING: Extract image patches of size (kH, kW) across all input channels C which are S stride(s) apart from each other. 
        This simulates the convolutional sliding window. Each patch is flattened into a 1D vector of shape P=(kH * kW * C, 0)
        # The operation then collects each patch vector into the re-shaped output tensor of shape [B, O, O, P]
        # where O stands for the size of the output activation filter. call_spiking function will be called for different patches in parallel.  

        For example, for a padded CIFAR10 input spike tensor with shape [B, 34, 34, 3] and a kernel size (3,3):
        As one patch also extracts the features from the channels, each patch will have a (flattened) shape of O = ((3*3) * 3) = (27) 
        We can extract 32 patches in the width direction and 32 in the height direction. 
        In total, all patches are collected in the output tensor with shape [B, 32, 32, 27]

        In order 

        ''' 


        # See: https://stackoverflow.com/a/75186655
        # tj = F.unfold(tj, self.kernel_size[0])

        patches = tj.unfold(2, kH, 1).unfold(3, kW, 1)  # [B, C, H_out, W_out, kH, kW]
        # Flatten the kernel dims
        patches = patches.contiguous().view(1, 3, 32, 32, -1)  # [1, 3, 32, 32, 9]

        # Merge channels and kernel into one dimension
        patches = patches.permute(0, 2, 3, 1, 4)
        patches = patches.reshape(1, 32, 32, -1)

        # config_utils.write_conv_tensor("torch_patched_tensor.txt", patches)

        # re-shape the weight to match its shape to a single patch P so that both input and weights can be passed to call_spiking 
        # for weight with 3 channels, kernel_size (kH, kW), filters F: [C, kH, kW, F] -> [P, F] 
        patch_W = self.kernel.reshape(-1, self.filters)        # flatten all dimensions except last one 


        tj = patches.reshape(-1, patch_W.shape[0])
        x = 0 
        y = 0
        ti = call_spiking(tj, patch_W, self.bias[0], self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        # Layer output is reshaped back.
        ti = ti.reshape(-1, image_same_size, image_same_size, self.filters)

        
        # breakpoint()
        # print("output return: ti.shape=", ti.shape)
        return ti

#### VERSION TO BE USED WITH THE h5-PREPROCESSED WEIGHTS FROM TENSORFLOW ####
class VGG_SNN_torch(nn.Module):
    def __init__(self, X_n, kernel_size, robustness_params, kernel_regularizer=None, kernel_initializer=None, dropout=0):
        super().__init__()
        self.X_n = X_n
        self.kernel_size = kernel_size
        self.name = "VGG_SNN_TORCH"

        # VGG_16 -> 
        layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']      
        layers1D = [512,10]

        self.features = nn.ModuleList()      # Conv blocks: [Conv -> ReLU -> MaxPool]
        self.classifier = nn.ModuleList()        # Fully connected

        image_size = 1
        prev_layer_dim = 3      # num channels for input image
        conv_index = 0
        pool_index = 0
        X_n_index = 0
        for filter in layers2D: 
            if filter != 'pool':
                X_n_layer = (X_n[X_n_index] if type(X_n)==list else X_n)
                layer_name = 'conv2d_' + str(conv_index+1)
                conv2d = SpikingConv2DTorch(filter, prev_layer_dim, X_n=X_n_layer, 
                                            padding='same', name=layer_name, robustness_params=robustness_params)
                if conv_index == 0:
                    conv2d.first_convolutional_layer = True 
                self.features.append(conv2d)
                conv_index += 1
                prev_layer_dim = filter 
                X_n_index += 1

            else: 
                ### Append the MinMaxPool custom layer to account for a sign change 
                if pool_index == 0: 
                    layer_name = f'max_min_pool2d'
                else: 
                    layer_name = f'max_min_pool2d_{pool_index}'
                max_min_pool = config_utils.MaxMinPool2d(name=layer_name)
                self.features.append(max_min_pool)
                image_size = image_size // 2
                pool_index += 1

        fc_index = 0
        dim_in = 512
        for dim_out in layers1D:
            X_n_layer = (X_n[X_n_index] if type(X_n)==list else X_n)
            layer_name = f'dense_{fc_index+1}'
            fc_layer = SpikingDenseTorch(dim_in, dim_out, X_n=X_n_layer, name=layer_name)
            self.classifier.append(fc_layer)
            dim_in = dim_out
            fc_index += 1
            X_n_index += 1

        self.classifier[-1].is_output = True 
        
        # Add one more dense layer for classification   
        # # TODO: pass number of classes as layer output dimension
        # self.fc_layer_out = SpikingDenseTorch(dim, 10, name="dense", X_n=X_n_layer, 
                                            #   is_output=True, robustness_params=robustness_params)

        # register hooks
        self.collect_activations = False
        self.all_activations = {}
        for i, layer in enumerate(self.features):
            if isinstance(layer, SpikingConv2DTorch):
                layer.register_forward_hook(self.get_activations(layer.name))
        
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, SpikingDenseTorch):
                layer.register_forward_hook(self.get_activations(layer.name))


    def forward(self, input_spikes):

        # print("\nVGG model call: tj.shape=", input_spikes.shape)

        x = input_spikes
        # breakpoint()
        for i, conv_block in enumerate(self.features):
            if isinstance(conv_block, SpikingConv2DTorch):
                # print(f"### Convolution {i}")
                x = conv_block(x)
            else: 
                # input to maxPool2d must be permuted again 
                # print("### MaxPool2d")

                x = -x 
                x = x.permute(0, 3, 1, 2)
                x = conv_block(x)
                x = x.permute(0, 2, 3, 1)   # permute back for convolution
                x = -x
        x = torch.flatten(x, 1)

        for layer in self.classifier:
            x = layer(x)

        return x
    def set_snn_intervals(self, t_min_start=0, t_max_start=1):
        # Helper function to create the [t_min, t_max] boundaries for the 
        #    integrate vs spike time windows for each layer. 
        #     't_min_start' and 't_max_start' define the min/max time values in the input layer. 
        # 
        t_min, t_max= t_min_start, t_max_start
        layer_num = 0

        for conv_layer in self.features:
            if isinstance(conv_layer, SpikingConv2DTorch):
                print("Setting SNN intervals in SpikingConv2DTorch")
                t_min, t_max = conv_layer.set_intervals(t_min, t_max)
            else: 
                print("Skipping because of maxpool")
        
        for fc_layer in self.classifier: 
            if isinstance(fc_layer, SpikingDenseTorch):
                print("Setting SNN intervals in SpikingDenseLayer")
                t_min, t_max = fc_layer.set_intervals(t_min, t_max)
    

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

    def dump_activations(self, path):
        '''
        Dumps all the layer-wise collected activations into the .npy file specified
        at 'path'. It also clears the dictionary 'self.activations' to make space for 
        later further collections. 
        '''
        with open(path, "wb") as f:
            pickle.dump(self.all_activations, f)
        self.all_activations = {}
    


'''   ##################################       VERSION FOR ANN-SNN conversion fully inside torch  ##############################
class VGG_SNN_torch(nn.Module):
    def __init__(self, X_n, kernel_size, robustness_params, kernel_regularizer=None, kernel_initializer=None, dropout=0):
        super().__init__()
        self.X_n = X_n
        self.kernel_size = kernel_size
        self.name = "VGG_SNN_TORCH"

        # VGG_19 -> 
        layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool']
        layers1D= [512, 512, 10]

        self.features = nn.ModuleList()      # Conv blocks: [Conv -> ReLU -> MaxPool]
        self.classifier = nn.ModuleList()        # Fully connected

        image_size = 1
        prev_layer_dim = 3      # num channels for input image
        conv_index = 0
        for filter in layers2D: 
            if filter != 'pool':
                X_n_layer = (X_n[conv_index] if type(X_n)==list else X_n)
                layer_name = 'conv2d_' + str(conv_index+1)
                conv2d = SpikingConv2DTorch(filter, prev_layer_dim, X_n=X_n_layer, 
                                            padding='same', name=layer_name, robustness_params=robustness_params)
                if conv_index == 0:
                    conv2d.first_convolutional_layer = True 
                self.features.append(conv2d)
                conv_index += 1
                prev_layer_dim = filter 

            else: 
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
                image_size = image_size // 2

        fc_index = 0
        dim_in = 512
        for dim_out in layers1D:
            X_n_layer = (X_n[fc_index] if type(X_n)==list else X_n)
            layer_name = f'dense_{fc_index+1}'
            fc_layer = SpikingDenseTorch(dim_in, dim_out, X_n=X_n_layer, name=layer_name)
            self.classifier.append(fc_layer)
            dim_in = dim_out
            fc_index += 1

        self.classifier[-1].is_output = True 


        # Add one more dense layer for classification   
        # # TODO: pass number of classes as layer output dimension
        # self.fc_layer_out = SpikingDenseTorch(dim, 10, name="dense", X_n=X_n_layer, 
                                            #   is_output=True, robustness_params=robustness_params)


    def forward(self, input_spikes):

        # print("\nVGG model call: tj.shape=", input_spikes.shape)

        x = input_spikes

        for i, conv_block in enumerate(self.features):
            if isinstance(conv_block, SpikingConv2DTorch):
                # print(f"### Convolution {i}")
                x = conv_block(x)
            else: 
                # input to maxPool2d must be permuted again 
                # print("### MaxPool2d")
                x = x.permute(0, 3, 1, 2)
                x = conv_block(x)
                x = x.permute(0, 2, 3, 1)   # permute back for convolution

        x = torch.flatten(x, 1)

        for layer in self.classifier:
            x = layer(x)

        return x
    def set_snn_intervals(self, t_min_start=0, t_max_start=1):
        # Helper function to create the [t_min, t_max] boundaries for the 
        #    integrate vs spike time windows for each layer. 
        #     't_min_start' and 't_max_start' define the min/max time values in the input layer. 
        # 
        t_min, t_max= t_min_start, t_max_start
        layer_num = 0

        for conv_layer in self.features:
            if isinstance(conv_layer, SpikingConv2DTorch):
                print("Setting SNN intervals in SpikingConv2DTorch")
                t_min, t_max = conv_layer.set_intervals(t_min, t_max)
            else: 
                print("Skipping because of maxpool")
        
        for fc_layer in self.classifier: 
            if isinstance(fc_layer, SpikingDenseTorch):
                print("Setting SNN intervals in SpikingDenseLayer")
                t_min, t_max = fc_layer.set_intervals(t_min, t_max)
'''


class VGG_ReLU_torch(nn.Module):
    def __init__(self, layers2D=[], kernel_size=(3,3), layers1D=[], BN=0, dropout=0, kernel_regularizer=None, kernel_initializer=None):
        super().__init__()

        self.kernel_size = kernel_size
        in_channels=3 
        num_classes=10

        layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D= [512]

        self.features = self._create_conv_layers(layers2D, in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def _create_conv_layers(self, filters, in_channels):

        layers = nn.ModuleList()
        for filter in filters:
            if filter == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, filter, kernel_size=3, padding=1)
                
                layers.append(nn.Sequential(
                    conv2d,
                    nn.BatchNorm2d(filter), 
                    nn.ReLU(inplace=True)
                ))
                in_channels = filter
        return layers

    def forward(self, x):
        x = x.float()
        for conv in self.features:
            x = conv(x)
        
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

'''
class VGG_ANN_torch(nn.Module):
    def __init__(self, features_list, batch_norm=False):
        super(VGG_ANN_torch, self).__init__()
        self.features = self.make_layers(features_list, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.collect_activations = False 
        self.max_activations = {}
        self.register_max_activation_forward_hooks()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, layer_architecture, batch_norm):
        layers = []
        in_channels = 3
        for v in layer_architecture:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def register_max_activation_forward_hooks(self):
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d) or isinstance(layer,config_utils.Conv2dWithBias):
                layer.register_forward_hook(self.get_max_activation(f"layer_{i}"))
        
        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(self.get_max_activation(f"layer_{i}"))

    def get_max_activation(self, name):
            Returns the maximum activation for a single layer during a forward pass.
            The max is updated for each individual batch (at each forward call). 
            Therefore, when predicting with a dataset, model.max_activations will store
            the maximum activation value across all batches in the training set.

            Function implementation also adapted from: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#using-the-forward-hooks (Accessed 27/03/25)

            @name: string, name of the layer
        def hook(model, input, output):
            relu_output = F.relu(output)
            batch_max = torch.max(relu_output).item()
            if name not in self.max_activations:
                self.max_activations[name] = batch_max
            else:
                self.max_activations[name] = max(self.max_activations[name], batch_max)

        return hook
'''
        
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
        self.name = 'FC_SNN_torch'
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
        from train_torch import evaluate_FC_SNN
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
   
class VGG(nn.Module):
    def __init__(self, architecture_list):
        super(VGG, self).__init__()

        layers_instance_list = self.vgg(architecture_list)

        self.features = nn.Sequential(*layers_instance_list)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10, bias=False)
        )

        for m in self.modules():
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        x = self.features(x)  
        x = x.view(-1, 512)
        x = self.classifier(x) 
        return x
    
    def vgg(self, cfg, i=3, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return layers


def create_torch_fc_model_ReLU(layers=2, N_hid=340,N_in=784, N_out=10):
    ''' Returns instance of a fully-connected ReLU model '''
    return FC_ReLU_torch(layers, N_hid, N_in, N_out)

def create_torch_fc_model_SNN(layers=2, N_hid=340, N_in=784, N_out=10, X_n=1000, robustness_params={}):
    ''' Returns an instance of a fully-connected SNN model '''
    return FC_SNN_torch(layers,N_hid,N_in,N_out,X_n,robustness_params=robustness_params, kernel_regularizer=None, kernel_initializer=None)

def create_torch_VGG_model_SNN(X_n, kernel_size):
    robustness_params = {'latency_quantiles': 1}
    return VGG_SNN_torch(X_n, kernel_size, robustness_params=robustness_params)

# def create_torch_VGG_model_ReLU():
#     return VGG_ReLU_torch()

def create_torch_VGG_model_ANN(features_list, batch_norm=False):
    
    VGG_16_features_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']            # vgg 16 
    VGG_19_features_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    model_instance = VGG_ANN_torch(VGG_19_features_list, batch_norm=False)
    model_instance = model_instance.double()
    return model_instance