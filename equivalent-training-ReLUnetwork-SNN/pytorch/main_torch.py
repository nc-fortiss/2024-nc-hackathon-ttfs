import argparse
from dataset_torch import Dataset_Torch
from model_torch import torch_fc_model_ReLU

override = None       # hard-code args parameters instead of passing them over the CLI

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
# print(args)
args = args[0]


''' 
    Instantiate objects with given parameters 
'''
args.model_name = args.data_name + '-' + args.model_name

robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'w_min': args.w_min,
    'w_max': args.w_max,
    'latency_quantiles':args.latency_quantiles
}

dataset = Dataset_Torch(
    args.data_name,
    flatten= ('FC' in args.model_name),
    convert_ttfs = ('SNN' in args.model_type),   
    ttfs_noise=args.noise,
)



nn = torch_fc_model_ReLU()
print(nn)

first_training_tuple = dataset.train_set[0]
first_tensor = first_training_tuple[0]
print(type(first_tensor))

res = nn.forward(first_tensor)
print(res)

