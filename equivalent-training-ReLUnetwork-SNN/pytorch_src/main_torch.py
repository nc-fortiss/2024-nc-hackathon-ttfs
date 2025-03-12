import argparse
from dataset_torch import Dataset_Torch
from train_torch import train_FC_SNN, evaluate_FC_SNN
import config_utils
from model_torch import *
import torch
from torch import nn
import pdb


override = None       # hard-code args parameters instead of passing them over the CLI

# Example run scripts, useful for testing 
# Train FC ReLU only:                  python3 main_torch.py --data_name=MNIST --model_type=ReLU --model_name=FC2
# Train FC SNN, save parameters:       python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2 --save=True 
# Load SNN model, evaluate test:       python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2 --load=True --testing=True --epochs=0
#
        
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
    model = create_torch_fc_model_SNN(layers=3, robustness_params=robustness_params)
elif 'ReLU' in args.model_type: 
    config_utils.logging.info("### Create instance of FC_ReLU: ###\n")
    model = create_torch_fc_model_ReLU(layers=3)

if model is None: 
    print('Please specify a valid model. Exiting.')
    exit(1)

print(model)
print("\n") 

''' Iterate over each hidden layer, plus the output layer, 
    and set the SNN interval time boundaries for each one. '''

if 'SNN' in args.model_type:
    config_utils.logging.info("### Setting SNNS intervals ####")
    t_min, t_max = 0, 1  
    layer_num = 0
    for child in model.children():
            config_utils.logging.info(child)
            
            if isinstance(child, nn.ModuleList):    # the hidden layers appear under a single child node as a moduleList
                for layer in child: 
                    t_min, t_max = layer.set_intervals(t_min, t_max)
                    config_utils.logging.info(f"layer_num={layer_num} -> B_n = {layer.B_n}; t_min_prev={layer.t_min_prev}; t_min={layer.t_min}; t_max={layer.t_max}\n")
                    layer_num += 1
            else: 
                t_min, t_max = child.set_intervals(t_min,t_max)    # for the output layer 
                config_utils.logging.info(f"layer_num={layer_num} -> B_n = {child.B_n}; t_min_prev={child.t_min_prev}; t_min={child.t_min}; t_max={child.t_max}\n")
                layer_num+=1 

''' Make a forward pass pre-training'''
config_utils.logging.info("--- Attempt forward pass ---")
model.eval()
tuple = dataset.train_set.__getitem__(0)
x = tuple[0]
config_utils.logging.info(f"Shape of input x: {(x.shape)}")
y = model(x)
config_utils.logging.info(f"Model output: {y}")

''' Make a test run on testset pre-training '''
if args.testing == True:
    config_utils.logging.info("--- Evaluating model on testset with no training ---")
    evaluate_FC_SNN(model, dataset.test_load)

''' Load pre-trained weights as needed '''
if args.load == True:
    # TODO: understand if it makes sense to save and load the full SNN weights ( - test accuracy drops immediately after loading)
    print("### Loading pre-trained weights ###")
    load_path = args.logging_dir + 'model/full_snn_weights.pth'
    model.load_state_dict(torch.load(load_path, weights_only=True))

''' Start training loop '''
if args.epochs > 0:
    config_utils.logging.info("--- Train the model: ---\n")
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-4)    # applying regularization (weight_decay) turns out to be crucial for training

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      # step-wise learning rate adjustment
    loss_fn = nn.CrossEntropyLoss()

    train_FC_SNN(model, dataset.train_load, args.epochs, optimizer=optimizer, scheduler=scheduler)
    config_utils.logging.info("--- Finished training the model ---")

    config_utils.logging.info("--- Evaluating model on testset ---")
    evaluate_FC_SNN(model, dataset.test_load)


''' Save model weights post-training'''
if args.save == True:
    # save the SNN when trained fully from scratch to avoid re-training (this is NOT the ANN-SNN conversion step)
    save_path = args.logging_dir + 'model/full_snn_weights.pth'
    config_utils.logging.info(f"Saving model post-training to {save_path}")
    torch.save(model.state_dict(), save_path)   # save weights as dict rather than the entire model

''' Make another forward pass post-training, log the input spike times and plot them '''
config_utils.logging.info("\n\n\n--- Attempt another forward pass ---\n")
config_utils.DEBUG_MODE = True
config_utils.clean_spike_logs() 
model.eval()
tuple = dataset.train_set.__getitem__(0)
x = tuple[0]
config_utils.logging.info(f"Shape of input x: {(x.shape)}")
y = model(x)
config_utils.logging.info(f"Model output: {y}")

config_utils.plot_input_spikes()


config_utils.DEBUG_MODE = False 

''' Evaluate the model on testset post-training '''
if args.testing == True:
    config_utils.logging.info("--- Evaluating model on testset post-training ---")
    evaluate_FC_SNN(model, dataset.test_load)



