import os 
import logging 
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils import fuse_conv_bn_eval
from collections import OrderedDict
import copy


'''
    Module containing global configuration settings and logging / utility functions
'''


DEBUG_MODE = False      # save further logging info for debugging if True
TRAIN_SHIFT = True      # re-train the SNN interval boundaries if True
LOGGING_DIR = ''

def set_up_logging(logging_dir, model_name):
    """
    Set up logging for the simulation. Taken from the tensorflow implementation
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
           logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
           logging.StreamHandler(sys.stdout),
        ],
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    
    # Set the logging_dir variable
    global LOGGING_DIR
    LOGGING_DIR = logging_dir

    # Create sub-directories if non-existent
    if not os.path.exists(logging_dir + '/outputs'):
        os.makedirs(logging_dir + '/outputs')

def write_conv_tensor(filename, tj):

    tj = tj.clone().detach().squeeze(0) 

    with open(filename, "w") as f:
        for c in range(tj.shape[0]):
            f.write(f"Channel {c}:\n")
            for row in tj[c]:
                row_str = ' '.join(f"{v.item():.3f}" for v in row)
                f.write(row_str + '\n')
            f.write("\n")
    
def clean_spike_logs(model):
    ''' Cleans any existing spike-time files from the logging directory  '''
    for file in os.listdir(LOGGING_DIR):
        if file.startswith('spike_output'):
            file_path = os.path.join(LOGGING_DIR, file)
            os.remove(file_path)  
            logging.info(f"### Removed {file_path} ###")

    model.layer_activations = [ np.empty([0]) for _ in range(len(model.hidden_layers)) ]


class Conv2dWithBias(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bias=False, **kwargs)
        self.register_buffer("BN", torch.tensor([0], dtype=torch.uint8))  # 0 or 1
        self.register_buffer("BN_before_ReLU", torch.tensor([0], dtype=torch.uint8))
        self.use_custom_bias = False

    def set_custom_bias(self, base_bias, W=None, b_term=[0.0]):
        """
            Creates per-location biases for handling padded boundaries during BN fusion.
            base_bias: [64, ]
            W: [64,3,3,3]      - in general [IN,OUT,H,W]
            b_term: [27, ]      - shorthand: b0 := b_term[0]
        """
        self.use_custom_bias = True
        out_channels = W.shape[0]
        self.custom_bias = nn.Parameter(torch.zeros(9, out_channels, dtype=torch.float64), requires_grad=False)

        if W is not None: 
            W_sum_2D = torch.sum(W, dim=(2, 3))     # sum over [H,W] => [64,3,3,3] -> [64,3]        # if matching tf,  W_sum_2D.ranspose(0, 1)
            b_term = b_term[:W.shape[1]]            # slice b_term from [27,] to [:OUT] dimension
        
        kernel = W
        for i in range(9):
            if i == 0:
                delta_sum_W = torch.zeros((b_term.shape[0], 1), dtype=torch.float64)        # [b0, 1]
            elif i == 1:
                delta_sum_W = W_sum_2D - kernel[:, :, 1:, 1:].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1)                # [3,64]

            elif i == 2:
                delta_sum_W = kernel[:, :, :1, :].sum(dim=(2, 3))       # [64,3]
                delta_sum_W = delta_sum_W.transpose(0,1)                # [3,64]

            elif i == 3:
                delta_sum_W = W_sum_2D - kernel[:, :, 1:, :-1].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1)                # [3,64]

            elif i == 4:
                delta_sum_W = kernel[:, :, :, -1:].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1)                # [3,64]

            elif i == 5:
                delta_sum_W = W_sum_2D - kernel[:, :, :-1, :-1].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1)         
                
            elif i == 6:
                delta_sum_W = kernel[:, :, -1:, :].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1)         

            elif i == 7:
                delta_sum_W = W_sum_2D - kernel[:, :, :-1, 1:].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1) 

            elif i == 8:
                delta_sum_W = kernel[:, :, :, :1].sum(dim=(2, 3))
                delta_sum_W = delta_sum_W.transpose(0,1) 


            diag_b = torch.diag(b_term)         # [b0, b0]
            mul = torch.matmul(diag_b, delta_sum_W)     # [b0,b0] * [b0, out]
            delta_bias = torch.sum(mul, dim=0)

            self.custom_bias[i] = base_bias - delta_bias

            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: break


    def forward(self, x):
        out = super().forward(x)
        if not self.use_custom_bias:
            return out + self.bias.view(1, -1, 1, 1) if self.bias is not None else out

        # Region-specific bias (9 regions)
        if self.padding != (1, 1) or self.BN.item() != 1 or self.BN_before_ReLU.item() == 1:
            return out + self.custom_bias[0].view(1, -1, 1, 1)

        out_ = torch.zeros_like(out)
        _, _, H, W = out.shape

        # Map region-wise bias
        out_[:, :, 1:-1, 1:-1] = out[:, :, 1:-1, 1:-1] + self.custom_bias[0].view(1, -1, 1, 1)
        out_[:, :, :1, :1] = out[:, :, :1, :1] + self.custom_bias[1].view(1, -1, 1, 1)
        out_[:, :, :1, 1:-1] = out[:, :, :1, 1:-1] + self.custom_bias[2].view(1, -1, 1, 1)
        out_[:, :, :1, -1:] = out[:, :, :1, -1:] + self.custom_bias[3].view(1, -1, 1, 1)
        out_[:, :, 1:-1, -1:] = out[:, :, 1:-1, -1:] + self.custom_bias[4].view(1, -1, 1, 1)
        out_[:, :, -1:, -1:] = out[:, :, -1:, -1:] + self.custom_bias[5].view(1, -1, 1, 1)
        out_[:, :, -1:, 1:-1] = out[:, :, -1:, 1:-1] + self.custom_bias[6].view(1, -1, 1, 1)
        out_[:, :, -1:, :1] = out[:, :, -1:, :1] + self.custom_bias[7].view(1, -1, 1, 1)
        out_[:, :, 1:-1, :1] = out[:, :, 1:-1, :1] + self.custom_bias[8].view(1, -1, 1, 1)

        return out_

class MaxMinPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)
        self.register_buffer("sign", None)  # shape: [1, C, 1, 1]

    def build(self, input_shape):
        channels = input_shape[1]
        self.sign = torch.ones((1, channels, 1, 1))

    def forward(self, x):
        if self.sign is None:
            # Initialize to all ones: shape[1,C,1,1]
            self.sign = torch.ones(1, x.shape[1], 1, 1, device=x.device, dtype=x.dtype)

        signed_input = self.sign * x
        pooled = self.max_pool(signed_input)
        return pooled * self.sign

def copy_layer(orig_layer): 
    if isinstance(orig_layer, nn.Conv2d):
        print("Converting Conv2D Layer to Conv2DBias")
        # Convert to Conv2dWithBias
        new_layer = Conv2dWithBias(
            in_channels=orig_layer.in_channels,
            out_channels=orig_layer.out_channels,
            kernel_size=orig_layer.kernel_size,
            stride=orig_layer.stride,
            padding=orig_layer.padding,
            dilation=orig_layer.dilation,
            groups=orig_layer.groups,
        )
        new_layer.weight.data.copy_(orig_layer.weight.data.clone())
        return new_layer

    elif isinstance(orig_layer, nn.MaxPool2d):
        # Convert to MaxMinPool2d (assuming similar interface)
        print("Converting MaxPool2D Layer to MaxMinPool2D")
        return MaxMinPool2d(kernel_size=orig_layer.kernel_size, stride=orig_layer.stride)
    else:
        return copy.deepcopy(orig_layer)

def copy_model(model, p, q):

    layer_index = 0 
    new_model = OrderedDict()
    if not (p==0 and q==1): layer_index = fuse_imaginary_bn_input(new_model, model, p, q)
    
    conv_layers = list(model.features.named_children())
    while layer_index < len(conv_layers):
        cur_layer = conv_layers[cur_layer]
        if isinstance(cur_layer, nn.Dropout):
            layer_index += 1
        fused_layer = copy_layer(cur_layer)

        if isinstance(cur_layer, nn.Conv2d):
            W = cur_layer.weight 
            b = cur_layer.bias 

            fused_layer.set_custom_bias(b)
        
        new_model[f"{layer_index}_layer"] = fused_layer

    

    return new_model         



def load_ANN_weights(snn_model, load_path):
    '''
        Load the trained ANN weights into an SNN instance.
        Since the model definitions are different (in particular the names for the layers and the weigths), 
        the loading requires manual adjustment; the standard model.load_weights() from torch cannot be used.

        @snn_model: a new SNN model instance
        @load_path: path to the .pth file with the ANN weights
    '''
    # Adapted from: https://discuss.pytorch.org/t/loading-weights-from-pretrained-model-with-different-module-names/11841/3 (Accessed 27/03/2025)
    ann_state_dict = torch.load(load_path, weights_only=True)
    snn_state_dict = snn_model.state_dict()

    layer=0
    for k,v in ann_state_dict.items():
        if 'weight' in k:
            layer_weight_key = f"output_layer.kernel" if layer == (snn_model.N_layers-1) else f"hidden_layers.{layer}.kernel"
            ANN_weights_tensor = v.T        # re-shape tensor to match SNN weights shapes
            snn_state_dict[layer_weight_key] = ANN_weights_tensor
            layer += 1

    # After collecting all the weights into the state_dict, load it into the SNN model
    snn_model.load_state_dict(snn_state_dict)

    ''' ### PLOT THE LOADED WEIGHTS
    layer_weights = []
    layer_names = []

    # Collect weights from each layer
    for name, param in snn_model.named_parameters():
        if 'weight' in name or 'kernel' in name:  # SNNs might use 'kernel'
            layer_weights.append(param.data.cpu().numpy().flatten())
            layer_names.append(name)


    plt.figure(figsize=(12, 6))
    for weights, name in zip(layer_weights, layer_names):
        plt.hist(weights, bins=100, alpha=0.5, label=name)


    plt.title("Weight Histograms per Layer")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''


def preprocess_relu(model):
    '''
        Preprocesses the pre-trained ReLU instance so that its weights 
        can be loaded into an SNN model instance (for further fine-tuning)
    
        @model: FC_ReLU_torch instance (trained)
    '''

    return 0

def convert_weights_tf_torch(tf_weights_path):
    '''
        Converts the weights from the VGG-16 tensorflow model to adapt them to 
        the torch model weight requirements. 
    '''
    return 0

def fuse_imaginary_bn_input(fused_model, orig_model_features, p, q):
    '''
        Fuses the input scaling ((x - p) / (q - p)) into the first layer's weights/bias 
        (avoiding an explicit normalization step).
    '''

    print("### Fusing Imaginary Input BN layer ###")
    first_layer = orig_model_features[0]  # First trainable layer
    with torch.no_grad():
        # For Conv2d
        if isinstance(first_layer, nn.Conv2d):
            # constant vector with shape of # input channels C: [C]
            kappa = (q - p) * torch.ones(first_layer.in_channels, dtype=torch.float64)       # e.g. [ 6.,  6.,  6.]
            b_term = p * torch.ones(first_layer.in_channels, dtype=torch.float64)            # e.g. [-3., -3., -3.]

            # extend the vector to account for kernel_size, so shape: [C] -> [C * H * W]
            kernel_size = first_layer.weight.shape[2:]  # (H, W)
            spatial_positions = kernel_size[0] * kernel_size[1]
            kappa_tiled = kappa.repeat(spatial_positions)  # [C*H*W, ]
            b_term_tiled = b_term.repeat(spatial_positions)  # [C*H*W, ]

            # Create diagonal matrix
            kappa_diag = torch.diag(kappa_tiled)  # [C*H*W, C*H*W]
            
            # Reshape kernel: [OUT, IN, H, W] -> [IN*H*W, OUT]
            W = first_layer.weight.view(first_layer.out_channels, -1).T  # [IN*H*W, OUT]
            
            # ---- Apply scaling to the weight (Eq. 13)
            W_fused = (kappa_diag @ W).T.view_as(first_layer.weight)  # [OUT, IN, H, W]

            
            # ---- Adjust bias (Eq. 12)
            # Original TF approach: Version 1: compute bias_adjustment := b_term_diag ([27,27]) @ W_flat ([64, 27])
            W_reshaped = first_layer.weight.permute(1, 2, 3, 0).reshape(-1, first_layer.out_channels)
            b_term_diag = torch.diag(b_term_tiled)          # [IN*H*W, IN*H*W]
            bias_adjustment = (b_term_diag @ W_reshaped).sum(dim=0)  # Sum over input dimensions
            b_fused = first_layer.bias + bias_adjustment

            # Transfer the fused weight and bias to the new fused layer 
            new_layer = copy_layer(first_layer)
            # Copy weights and set BN flags
            new_layer.weight.data.copy_(W_fused)
            new_layer.BN.fill_(1)          # Set BN flag to 1
            new_layer.BN_before_ReLU.fill_(0)  # Set BN_before_ReLU to 0

            # Set custom bias (handles padding)
            # TODO: pay attention to whether passing b_term_tiled or b_term as in tf b_term might get re-assigned after if 'conv'
            new_layer.set_custom_bias(b_fused, W=first_layer.weight.data.clone(), b_term=b_term_tiled)      

            # --- Save the new fused layer in the model 
            fused_model[f"{0}_conv_bn"] = new_layer 
            
            # --- Already save the next ReLU activation layer from the original model
            next_layer_activation = copy_layer(orig_model_features[1])
            assert(isinstance(next_layer_activation, nn.ReLU))
            fused_model[f"{1}_activation"] = next_layer_activation

        else:
            print("!!! Cannot apply batch norm at first layer ")      
            exit(1)      

    # return i=3 so that we can skip the 
    return 3

def fuse_bn_after_activation(fused_model, orig_model_features, i):
    breakpoint()
    bn_layer = orig_model_features[i]
    assert isinstance(bn_layer, nn.BatchNorm2d)

    # Step 1: Compute kappa and b_term
    gamma = bn_layer.weight           # shape: [C]
    beta = bn_layer.bias              # shape: [C]
    mean = bn_layer.running_mean      # shape: [C]
    var = bn_layer.running_var        # shape: [C]
    eps = bn_layer.eps                # scalar

    kappa = gamma / torch.sqrt(var + eps)
    b_term = beta - mean * kappa

    # Step 2: skip dropout layers
    i += 1      
    while i < len(orig_model_features) and isinstance(orig_model_features[i], nn.Dropout):     
        i += 1

    # Step 3: Replace MaxPool2d with MaxMinPool2d and change sign
    if isinstance(orig_model_features[i + 1], nn.MaxPool2d):
        print("Replacing MaxPool2d with MaxMinPool2d")
        mp = orig_model_features[i + 1]
        mmp = MaxMinPool2d(kernel_size=mp.kernel_size, stride=mp.stride)

        # Initialize mmp.sign based on kappa's sign
        # Assume kappa shape is [C] for now
        sign = torch.sign(kappa).view(1, -1, 1, 1)  # [1, C, 1, 1]
        mmp.sign = sign  # or use mmp.register_buffer("sign", sign.clone())

        fused_model[f"{i}_mmp"] = mmp
        i += 1

        # Skip dropout if it follows
        if i + 1 < len(orig_model_features) and isinstance(orig_model_features[i + 1], nn.Dropout):
            i += 1

    # TODO: pass flatten layer and handle case 

    # Step 4: Fuse into the next parametrized layer (Conv or Linear)

    if (i + 1) < len(orig_model_features) and isinstance(orig_model_features[i+1], nn.Conv2d):
        next_layer = orig_model_features[i+1]            # TODO: is this (i) or (i+1) ?
        W = next_layer.weight.data  # shape: [out_ch, in_ch, kH, kW]
        out_channels, in_channels, kH, kW = W.shape

        # Expand kappa/b_term to match flattened shape of W
        kappa_exp = kappa.repeat_interleave(kH * kW)  # [C*kH*kW]
        b_term_exp = b_term.repeat_interleave(kH * kW)

        # Flatten W to [C*kH*kW, out_ch]
        W_flat = W.view(out_channels, -1).t()  # shape: [in*C*kH*kW, out]

        # Fuse: W_fused = diag(kappa) @ W
        W_fused = (kappa_exp.view(-1, 1) * W_flat).t().contiguous()
        W_fused = W_fused.view_as(W)

        # Fuse bias
        if next_layer.bias is None:
            next_layer.bias = nn.Parameter(torch.zeros(out_channels))
        b_fused = next_layer.bias.data + torch.matmul(b_term_exp.view(1, -1), W_flat).view(-1)

        # Replace with Conv2DWithBias
        fused_conv = Conv2dWithBias(
            in_channels=next_layer.in_channels,
            out_channels=next_layer.out_channels,
            kernel_size=next_layer.kernel_size,
            stride=next_layer.stride,
            padding=next_layer.padding,
            dilation=next_layer.dilation,
            groups=next_layer.groups,
        )
        fused_conv.weight.data.copy_(W_fused)
        fused_conv.bias = nn.Parameter(b_fused)
        fused_conv.BN[...] = 1
        fused_conv.BN_before_ReLU[...] = 0

        fused_model[f'{i}_fused'] = fused_conv
        i += 1

    elif (i + 1) < len(orig_model_features) and isinstance(orig_model_features[i+1], nn.Linear):
        # Similar fusion logic for Linear
        W = next_layer.weight.data  # [out, in]
        W_fused = kappa.view(1, -1) * W
        b_fused = next_layer.bias.data + b_term.view(1, -1) @ W.t()

        next_layer.weight.data.copy_(W_fused)
        next_layer.bias.data.copy_(b_fused.view(-1))
        fused_model[f'{i}_fused'] = next_layer
        i += 1

    return i+2 

def remove_dropout(module):
    """Replace all Dropout layers with Identity."""
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            print(f"Removing dropout: {name}")
            setattr(module, name, nn.Identity())
        else:
            remove_dropout(child)
    return module

def fuse_bn(model, p, q, BN = True, BN_before_ReLU = False, layer_idx=[0]):
    """
    Creates new models which:
        Fuses all (imaginary) batch normalization layers; 
        Changes bias on locations where it is needed; 
        Transforms MaxPooling layers in MaxMinPooling layers and Conv2D layers in Conv2DWithBias.  
    """
    logging.info("## Fusing BN layers ###")
    new_layers = OrderedDict()
    i = 0

    ## STANDARD BATCH NORM on input layer ### 
    if not (p==0 and q==1):
        logging.info("## Simulate a BN layer to scale data ###")
        fuse_imaginary_bn_input(new_layers, model.features, p, q)
        
    print(new_layers)
    if BN: 
        # First apply BN to the convolutional layers
        for i, layer in enumerate(model.features):
            if (p==0 and q==1) and i==1:
                pass    # TODO 
            
            if isinstance(layer, nn.BatchNorm2d):
                print("### fusing bn after activations ###")
                i = fuse_bn_after_activation(new_layers, model.features, i)


    else: 
        pass        # TODO

        # if isinstance(layer, nn.Conv2d):
        #     # replace Conv2D with Conv2DWithBias

        #     conv = Conv2dWithBias(
        #             in_channels=layer.in_channels,
        #             out_channels=layer.out_channels,
        #             kernel_size=layer.kernel_size,
        #             stride=layer.stride,
        #             padding=layer.padding,
        #             # dilation=layer.dilation,
        #             # groups=layer.groups,
        #     )
        #     conv.weight.data.copy_(layer.weight.data.clone())
        #     if layer.bias is not None:
        #         conv.bias = nn.Parameter(layer.bias.data.clone())

    for k, v in new_layers.items():
        print(f"k={k} - v={v}")        
    return new_layers 

def insert_batchnorm_after_conv(sequential_convolutions):
    print("\n\n Inserting BatchNorm Layers for convolutional layers")
    conv_layers = list(sequential_convolutions.named_children())
    new_conv_layers = OrderedDict() 
    layer_index = 0
    for i, (name, layer) in enumerate(conv_layers):
        new_conv_layers[f"{layer_index}_{name}"] = layer
        layer_index += 1

        if isinstance(layer, nn.ReLU):
            # Look backward to find the preceding Conv2d for channel count
            prev_conv = None
            for j in range(i - 1, -1, -1):
                prev = conv_layers[j][1]
                if isinstance(prev, nn.Conv2d):
                    prev_conv = prev
                    break

            if prev_conv is None:
                raise ValueError(f"Cannot find Conv2d before ReLU at layer {i}")

            # Add BatchNorm2d and Dropout
            bn = nn.BatchNorm2d(prev_conv.out_channels)
            bn.bias.data = bn.bias.data.double()
            do = nn.Dropout2d(p=0.1)

            new_conv_layers[f"{layer_index}_bn"] = bn
            layer_index += 1
            new_conv_layers[f"{layer_index}_dropout"] = do
            layer_index += 1



        # if isinstance(layer, nn.Conv2d):
        #     # Insert BatchNorm2d immediately after Conv2d
        #     bn = nn.BatchNorm2d(layer.out_channels)
        #     bn.bias.data = bn.bias.data.double()        # convert parameters to float64
        #     new_conv_layers[f"{layer_index}_bn"] = bn
        #     layer_index += 1
    return nn.Sequential(new_conv_layers)

def insert_batch_norm_after_linear(sequential_linear): 
    flat_layers = sequential_linear.named_children()
    new_layers = OrderedDict()
    idx = 0

    for name, layer in flat_layers:
        new_layers[f"{idx}_{name}"] = layer
        idx += 1
        if isinstance(layer, nn.Linear):
            bn = nn.BatchNorm1d(layer.out_features)
            new_layers[f"{idx}_bn"] = bn
            idx += 1

    return nn.Sequential(new_layers)

def insert_batch_norm(model):
    print("Inserting BN layers")
    model.features = insert_batchnorm_after_conv(model.features)
    print(model.features)
    model.classifier = insert_batch_norm_after_linear(model.classifier)
    print(model.classifier)
    return model
    
def convert_model(model, p,q):
    layers = []
    modules = list(model.features.children())

    i = 0

    layers.append(copy_layer(modules[0]))
    i = 1


    # Step 2: Apply imaginary BN if input range ≠ [0, 1]
    if not (p == 0 and q == 1):
        fuse_imaginary_bn_input(layers[0], p, q)

    while i < len(modules):
        current = modules[i]

def preprocess_relu(model, p, q, batch_normalization=True):
    model = model.eval()

    if batch_normalization: 

        # Since the pre-trained model does not contain any batch normalization layers, we add them here
        insert_batch_norm(model)
        print("\n\nNew batched model:")
        print(model)
        fuse_bn(model, p, q)

    else: 
        print("Create a copy of the model with no batch normalization")
        copy_model(model, p, q)


    for i, layer in enumerate(model.features):
        print(f"{i} - {layer}")
    

    # if batch_normalization:
    #     model = fuse_bn(model.features, p, q)
    #     print("Remove dropout")
    #     model = remove_dropout(model)

    return model 