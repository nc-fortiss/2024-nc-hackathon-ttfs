import os 
import logging 
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
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