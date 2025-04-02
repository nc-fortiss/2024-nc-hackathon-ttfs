import argparse
from dataset_torch import Dataset_Torch
from train_torch import train_FC_SNN, evaluate_FC_SNN
import config_utils
from model_torch import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle as pkl
import pdb
import os 
import plotting


override = None       # hard-code args parameters instead of passing them over the CLI

# Example run scripts, useful for testing 
# Train FC ReLU only:                           python3 main_torch.py --data_name=MNIST --model_type=ReLU --model_name=FC2 --epochs=1 --save=True
# Train FC SNN only (no conversion):            python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2 --epochs=5
# Convert ANN-SNN, evaluate test with no train: python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2 --load=True --testing=True --epochs=0
# Fine-tune SNN on ANN weights, evaluate test:  python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2 --load=True --testing=True --epochs=1
        
'''
    Command-line argument parsing
'''
# Architecture setup
strtobool = (lambda s: s=='True')       # For parsing boolean CLI args parameters
parser = argparse.ArgumentParser(description='TTFS')
parser.add_argument('--data_name', type=str, default='MNIST', help='(MNIST|CIFAR10|CIFAR100)')                              # name of the dataset to use
parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')         
parser.add_argument('--model_type', type=str, default='SNN', help='(SNN|ReLU)')                                             # choose between SNN and ReLU                              
parser.add_argument('--model_name', type=str, default='FC2', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
parser.add_argument('--layers', type=int, default='2', help='number of layers for a FC model')
parser.add_argument('--train_shift', type=strtobool, default=True, help='Re-calculate interval boundaries during training.')

# Hyperparameters
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Epochs. 0 -skip training')

# Train, test and load modes
parser.add_argument('--testing', type=strtobool, default=True, help='Execute testing.')
parser.add_argument('--load', type=str, default='False', help='Load before training. (True|False|custom_name.h5)')          # Load or save the trained network weights
parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')

# Robustness parameters:
parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument('--mode', type=str, default='', help='Ignore: A hack to address a bug in argsparse during debugging')

# Returns a tuple: args[0] is a Namespace with all the known parameters from 'parser' and 'override', args[1] contains ignored unknown parameters 
args = parser.parse_known_args(override)
if(len(args[1])>0):
    print("Warning: Ignored args", args[1])
print("Argument parameters: \n", args[0])
args = args[0]

''' 
    Instantiate objects with given parameters 
'''
args.model_name = args.data_name + '-' + args.model_name
config_utils.TRAIN_SHIFT = args.train_shift
config_utils.set_up_logging(args.logging_dir, args.model_name)
robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'w_min': args.w_min,
    'w_max': args.w_max,
    'latency_quantiles':args.latency_quantiles
}

''' Instantiate Data Loaders with appropriate parameters'''
dataset = Dataset_Torch(
    args.data_name,
    flatten= ('FC' in args.model_name),
    convert_ttfs = ('SNN' in args.model_type),   
    ttfs_noise=args.noise,
)

''' Instantiate model '''

model = None 
if 'SNN' in args.model_type:
    config_utils.logging.info("### Create instance of FC_SNN: ###\n")
    model = create_torch_fc_model_SNN(X_n=10, layers=args.layers, robustness_params=robustness_params)
elif 'ReLU' in args.model_type: 
    config_utils.logging.info("### Create instance of FC_ReLU: ###\n")
    model = create_torch_fc_model_ReLU(layers=args.layers)

if model is None: 
    print('Please specify a valid model. Exiting.')
    exit(1)

print(model)
print("\n") 


''' Load pre-trained weights as needed (pass --load=True or --load==custom_name)'''
if args.load != 'False':
    config_utils.logging.info("### Loading weights ###")
    if 'ReLU' in args.model_type:
        # Load weights
        if args.load == 'True':  # automatic name
            model.load_weights(args.logging_dir + args.model_name + '_weights.h5', by_name=True)
        else:  # custom name
            model.load_weights(args.logging_dir + args.load, by_name=True)
    if 'SNN' in args.model_type:
        # Load X_n ranges from pre-trained ANN, if available
        if os.path.exists(args.logging_dir + args.model_name + '_X_n.pkl'):
            X_n = pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))
        else:
            X_n = [10,50] 
        if 'FC2' in args.model_name:
            config_utils.logging.info(f"### Create new SNN model instance with loaded X_n = {X_n} ###\n")
            model = create_torch_fc_model_SNN(X_n=X_n, layers=args.layers, robustness_params=robustness_params)

        # After creating model instance with new X_n ranges, load the weights
        load_path = args.logging_dir + args.model_name + '_weights.pth'
        config_utils.load_ANN_weights(model, load_path)


    
''' Iterate over each hidden layer, plus the output layer, 
    and set the SNN interval time boundaries for each one. '''

if 'SNN' in args.model_type:
    config_utils.logging.info("### Setting SNN intervals ####")
    model.set_snn_intervals(0,1)

    config_utils.logging.info("### Print layer interval boundaries BEFORE training ###")
    for n, layer in enumerate(model.hidden_layers):
        config_utils.logging.info(f"layer_{n}: t_min={layer.t_min}, t_max={layer.t_max}")
    config_utils.logging.info(f"output_layer: t_min={model.output_layer.t_min}, t_max={model.output_layer.t_max}\n")

''' Make a forward pass pre-training'''
config_utils.logging.info("--- Attempt forward pass ---")
model.eval()
tuple = dataset.train_set.__getitem__(0)
x = tuple[0]
config_utils.logging.info(f"Shape of input x: {(x.shape)}")
y = model(x)
config_utils.logging.info(f"Model output: {y}")#


''' Make a test run on testset pre-training '''
if args.testing == True:
    config_utils.logging.info("\n--- Evaluating model on testset before training // fine-tuning ---")
    evaluate_FC_SNN(model, dataset.test_load)


''' Start training loop '''
if args.epochs > 0:
    config_utils.logging.info("--- Train the model: ---\n")
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-4)    # applying regularization (weight_decay) turns out to be crucial for training

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      # step-wise learning rate adjustment
    loss_fn = nn.CrossEntropyLoss()

    if 'SNN' in args.model_type:
        train_FC_SNN(model, dataset.train_load, args.epochs, optimizer=optimizer, scheduler=scheduler)
    elif 'ReLU' in args.model_type:
        model.fit(dataset.train_load, optimizer, loss_fn, epochs=args.epochs)
    
    config_utils.logging.info("--- Finished training the model ---")

if args.testing and args.epochs > 0:
    config_utils.logging.info("### Final test set accuracy (after training / fine-tuning) ###")
    evaluate_FC_SNN(model, dataset.test_load)    
    
''' Save model weights post-training'''
if args.save == True:
    if 'ReLU' in args.model_type:
        config_utils.logging.info("\n\n#### Saving ReLU model ####")
        # Save raw ANN weights (these can be used only for new ANN instances)
        save_path = args.logging_dir + args.model_name + '_weights.pth'
        torch.save(model.state_dict(), save_path) 

        # Preprocess ANN weights so that they can be used for the SNN conversion
        # TODO ? 

        # Save the optimal X_n ranges as the maximum ReLU activations
        config_utils.logging.info(f"### Maximum layer-wise ReLU activations: {model.max_activations}")
        X_n_list = [v for v in model.max_activations.values()]
        pkl.dump(X_n_list, open(args.logging_dir + '/' + args.model_name + '_X_n.pkl', 'wb'))

    if 'SNN' in args.model_type:
        # save the SNN when trained fully from scratch to avoid re-training (this is NOT the ANN-SNN conversion step)
        save_path = args.logging_dir + 'model/full_snn_weights.pth'
        config_utils.logging.info(f"Saving model post-training to {save_path}")
        torch.save(model.state_dict(), save_path)   # save weights as dict rather than the entire model

if 'SNN' in args.model_type:
    config_utils.logging.info("\n### Get layer time intervals AFTER training ###")
    for n, layer in enumerate(model.hidden_layers):
        config_utils.logging.info(f"layer_{n}: t_min={layer.t_min}, t_max={layer.t_max}")
    config_utils.logging.info(f"output_layer: t_min={model.output_layer.t_min}, t_max={model.output_layer.t_max}\n")


''' Make another forward pass post-training, log the input spike times and plot them '''
config_utils.logging.info("\n\n\n--- Attempt another forward pass ---\n")
config_utils.DEBUG_MODE = True          # this will enable the logging+print statements in each forward pass call
config_utils.clean_spike_logs() 
model.eval()
tuple = dataset.train_set.__getitem__(0)
x = tuple[0]
config_utils.logging.info(f"Shape of input x: {(x.shape)}")
y = model(x)
config_utils.logging.info(f"Model output: {y}")
print(model.layer_activations)
print(" \n")



''' -------------------- Membrane Potential Plots --------------------- '''

# Get the index of the neuron that produced the very first spike in the next layer
output_activations = model.layer_activations[1]
sorted_output_activations = np.sort(model.layer_activations[1])
sorted_index_activations = np.argsort(model.layer_activations[1])
min_spike_neuron_index = sorted_index_activations[0]


config_utils.plot_input_spikes()
# plotting.plot_membrane_potential(model, [min_spike_neuron_index, 140, 200])
plotting.plot_membrane_potential(model, [min_spike_neuron_index])