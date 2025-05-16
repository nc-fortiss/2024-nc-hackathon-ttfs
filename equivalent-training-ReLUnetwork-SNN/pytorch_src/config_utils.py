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

def remove_dropout(module):
    """Replace all Dropout layers with Identity."""
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            print(f"Removing dropout: {name}")
            setattr(module, name, nn.Identity())
        else:
            remove_dropout(child)
    return module



def fuse_bn(module, p, q, optimizer, BN = True, BN_before_ReLU = False):
    """
    Creates new models which:
        Fuses all (imaginary) batch normalization layers; 
        Changes bias on locations where it is needed; 
        Transforms MaxPooling layers in MaxMinPooling layers and Conv2D layers in Conv2DWithBias.  
    """
    logging.info("## Fusing BN layers ###")
    
    if not (p==0 and q==1):
        logging.info("## Simulate a BN layer to scale data ###")

    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            for i in range(len(child) - 1):
                if isinstance(child[i], nn.Conv2d) and isinstance(child[i + 1], nn.BatchNorm2d):
                    print(f"Fusing Conv+BN in {name}[{i}]")
                    fused = fuse_conv_bn_eval(child[i], child[i + 1])
                    child[i] = fused
                    child[i + 1] = nn.Identity()
        else:
            fuse_bn(child)
    return module


    fused_model = tf.keras.Sequential()
    # Add input layer.
    fused_model.add(copy_layer(model.layers[0]))
    i=1
    # If condition is satisfied, there is an imaginary batch normalization layer which is merged.
    if not (p==0 and q==1): i = fuse_imaginary_bn(fused_model, model, p, q)
    if BN:
        # There are batch normalization layers.
        if BN_before_ReLU:
            # Batch normalization layers are always found before ReLU activation function.
            while i<len(model.layers):
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with previous convolutional or fully-connected layer. 
                    i = fuse_bn_before_activation(fused_model, model, i)
                if i==(len(model.layers)-1):
                    # Add last Dense layer with 'softmax'.
                    layer = copy_layer(model.layers[-2])
                    layer.set_weights(model.layers[-2].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[-1]))
                i+=1
        else:
            # Batch normalization layers are always found after ReLU activation function.
            while i<len(model.layers): 
                # Add first Dense or Convolutional layer if there was no imaginary batch normalization.
                if (p==0 and q==1) and i==1:
                    layer = copy_layer(model.layers[1])
                    if 'conv' in model.layers[1].name:
                        kernel, bias = model.layers[1].get_weights()
                        # Set BN flag to 0 and BN_before_ReLU to 0.
                        layer.set_weights([kernel, np.array([0]), np.array([0])])
                        # Bias is same for all locations.
                        layer.set_bias(bias)
                    else:
                        layer.set_weights(model.layers[1].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[2]))
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with next convolutional or fully-connected layer.
                    i = fuse_bn_after_activation(fused_model, model, i)
                i+=1
    else:
        print("Create copy model")
        # If there is no batch normalization layers, copy model such that Conv2D and MaxPooling layers are replaced with ConvWithBias and MaxMinPooling respectively. 
        copy_model(fused_model, model, i)
    fused_model.compile(metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=optimizer)  
    return fused_model



def preprocess_relu(model, p, q, batch_normalization=True):
    model = model.eval()

    if batch_normalization:
        model = fuse_bn(model, p, q)
        model = remove_dropout(model)

    return model 