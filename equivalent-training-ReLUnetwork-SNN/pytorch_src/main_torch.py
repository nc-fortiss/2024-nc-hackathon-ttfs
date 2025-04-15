import argparse
from dataset_torch import Dataset_Torch
import train_torch
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
parser.add_argument('--latency_quantiles', type=float, default=1.0, help='Number of quantiles to take into account when calculating t_max. default no crop')
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
    args.batch_size,
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

config_utils.logging.info(model)
config_utils.logging.info("\n") 


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
        if os.path.exists(args.logging_dir + 'full_snn_weights.pth'):
            config_utils.logging.info("### Loading full SNN weights")
            load_path = args.logging_dir + 'full_snn_weights.pth'
            args.testing=True
        else: 
            config_utils.logging.info("### Loading converted ANN weights")
            load_path = args.logging_dir + args.model_name + '_weights.pth'
        
        model.load_state_dict(torch.load(load_path, weights_only=True))
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
config_utils.logging.info(f"Model output: {y}")

image = x.view(28, 28)

# Convert to numpy and plot
plt.imshow(image.numpy(), cmap='gray')
plt.title("28x28 Image from Flattened Tensor")
plt.axis('off')
plt.show()
plotting.plot_input_tensor(image)


''' Make a test run on testset pre-training '''
if args.testing == True:
    config_utils.logging.info("\n--- Evaluating model on testset before training // fine-tuning ---")
    if 'SNN' in args.model_type:
        train_torch.evaluate_FC_SNN(model, dataset.test_load)
    elif 'ReLU' in args.model_type:
        train_torch.evaluate_FC_ReLU(model, dataset.test_load)
    else:
        config_utils.logging.info("###  Invalid model ###")
        exit(1)


''' Start training loop '''
if args.epochs > 0:
    config_utils.logging.info("--- Train the model: ---\n")
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=1e-4)    # applying regularization (weight_decay) turns out to be crucial for training

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      # step-wise learning rate adjustment
    loss_fn = nn.CrossEntropyLoss()

    if 'SNN' in args.model_type:
        train_torch.train_FC_SNN(model, dataset.train_load, args.epochs, optimizer=optimizer, scheduler=scheduler)
    elif 'ReLU' in args.model_type:
        train_torch.train_FC_ReLU(model, dataset.train_load, optimizer, loss_fn, epochs=args.epochs)
    
    config_utils.logging.info("--- Finished training the model ---")

if args.testing and args.epochs > 0:
    config_utils.logging.info("### Final test set accuracy (after training / fine-tuning) ###")
    if 'SNN' in args.model_type:
        train_torch.evaluate_FC_SNN(model, dataset.test_load)
    elif 'ReLU' in args.model_type:
        train_torch.evaluate_FC_ReLU(model, dataset.test_load)
   
    
''' Save model weights post-training'''
if args.save == True:
    if 'ReLU' in args.model_type:
        config_utils.logging.info("\n\n#### Saving ReLU model ####")
        # Save raw ANN weights (these can be used only for new ANN instances)
        save_path = args.logging_dir + args.model_name + '_weights.pth'
        torch.save(model.state_dict(), save_path) 

        # Preprocess ANN weights so that they can be used for the SNN conversion (-- only relevant for VGG model)
        # TODO ? 

        # Save the optimal X_n ranges as the maximum ReLU activations
        config_utils.logging.info(f"### Maximum layer-wise ReLU activations: {model.max_activations}")
        X_n_list = [v for v in model.max_activations.values()]
        pkl.dump(X_n_list, open(args.logging_dir + '/' + args.model_name + '_X_n.pkl', 'wb'))

    if 'SNN' in args.model_type:
        # save the SNN when trained to avoid re-training (this is NOT the ANN-SNN conversion step)
        save_path = args.logging_dir + 'full_snn_weights.pth'
        config_utils.logging.info(f"### Saving model post-training to {save_path}")
        torch.save(model.state_dict(), save_path)   # save weights as dict rather than the entire model

if 'SNN' in args.model_type:
    config_utils.logging.info("\n### Get layer time intervals AFTER training ###")
    for n, layer in enumerate(model.hidden_layers):
        config_utils.logging.info(f"layer_{n}: t_min={layer.t_min}, t_max={layer.t_max}")
    config_utils.logging.info(f"output_layer: t_min={model.output_layer.t_min}, t_max={model.output_layer.t_max}\n")


''' Make another forward pass post-training, log the input spike times and plot them '''
# config_utils.logging.info("\n\n\n--- Attempt another forward pass ---\n")
# model.eval()
# tuple = dataset.train_set.__getitem__(0)
# x = tuple[0]
# config_utils.logging.info(f"Shape of input x: {(x.shape)}")
# y = model(x)
# config_utils.logging.info(f"Model output: {y}")

if 'SNN' in args.model_type and model.N_layers >= 3:
    ''' ------ Save activations from the above forward pass -------- '''
    model.collect_activations = True 
    model.eval()
    tuple = dataset.train_set.__getitem__(0)
    x = tuple[0]
    y = model(x)

    output_activations = model.activations['layer_1']
    sorted_index_activations = np.argsort(output_activations)
    min_spike_neuron_index = sorted_index_activations[0]
    dump_path = args.logging_dir + 'outputs/' + args.model_name + '_pass.npz'
    model.dump_activations(dump_path)

    
    plotting.plot_membrane_potential_path(model, dump_path, [min_spike_neuron_index, 40,200], title_addition='Forward Pass - Unoptimized')
    plotting.plot_output_spikes(dump_path, model=model, additional_title='\nForward Pass - Unoptimized')
    
    model.collect_activations = False 
    print("\n### Accuracy without optimizations: ")
    train_torch.evaluate_FC_SNN(model, dataset.test_load)
    print(f"output_layer.t_min={model.output_layer.t_min}\n\n")
    


    ''' ----------  Apply latency quantiles optimization  ----------'''
    # # Apply latency quantiles aat 99%, ..., 95% - evaluate and save activations
    # for q in reversed(range(95,100)):
    #     config_utils.logging.info(f"### Apply latency quantile q={q}%")
    #     model.apply_max_quantiles(q)

    #     #train_torch.evaluate_FC_SNN(model, dataset.test_load)
    #     y = model(x)
    #     optimization_name = f'_{q}.npz'
    #     optimized_activations_path = args.logging_dir + 'outputs/' + args.model_name + optimization_name
    #     model.dump_activations(optimized_activations_path)

    #     for n, layer in enumerate(model.hidden_layers):
    #         config_utils.logging.info(f"layer_{n}: t_max={layer.t_max}, t_max_q={layer.robustness_params['latency_quantiles'] * layer.t_max}")
    
    # # Plot activations after optimizations
    # for q in reversed(range(95,100)):
    #     optimization_name = f'_{q}.npz'
    #     optimized_activations_path = args.logging_dir + 'outputs/' + args.model_name + optimization_name
    #     plotting.plot_output_spikes(optimized_activations_path, model=model, additional_title=f'\n{str(q)}')


    ''' ---------- Apply threshold optimization -------- '''
    config_utils.logging.info("### Apply threshold adjustment optimization")
    model.optimize_threshold(dataset.test_load)
    config_utils.logging.info(f"### Model min_spike_times={model.min_spike_times}")
    config_utils.logging.info("### Get intervals after optimizing")
    for i, layer in enumerate(model.hidden_layers):
        print(f"layer_{i}: t_min={layer.t_min} -- t_max={layer.t_max}")

    # config_utils.DEBUG_MODE = True
    model.collect_activations = True
    y = model(x)
    optimized_path = args.logging_dir + 'outputs/' + args.model_name + '_threshold.npz'
    model.dump_activations(optimized_path)
    print(model.activations)


    # breakpoint()
    plotting.plot_membrane_potential_path(model, optimized_path, [min_spike_neuron_index, 40,200], title_addition='Forward Pass - Threshold Opt.')
    plotting.plot_output_spikes(optimized_path, model=model,additional_title='\nForward Pass - Threshold Opt.')

    # Test accuracy again after optimizing
    model.collect_activations = False   
    print("\nAccuracy with threshold adjustment")    
    evaluate_FC_SNN(model, dataset.test_load)
    print(f"output_layer.t_min={model.output_layer.t_min}\n\n")


    
    ''' ------------ Reduce t_max ---------------'''
    model.apply_max_quantiles(100)             # reset latency quantile

    new_t_max = 1
    for layer in model.hidden_layers:
        print(f"current t_max={layer.t_max}, current t_min={layer.t_min}")
        layer.t_min = new_t_max
        new_t_max = 0.1 * layer.t_max 
        layer.t_max = new_t_max
        print(f"updated t_max={layer.t_max}, updated t_min={layer.t_min}")
    model.output_layer.t_min=new_t_max
    

    model.collect_activations = True
    y = model(x)
    dump_path = args.logging_dir + 'outputs/' + args.model_name + '_shifted.npz'
    model.dump_activations(dump_path)

    plotting.plot_membrane_potential_path(model, dump_path, [min_spike_neuron_index, 40,200], title_addition='Forward Pass - Shifted t_max')
    plotting.plot_output_spikes(dump_path, model=model, additional_title='\nForward Pass - Shifted t_max')

    model.collect_activations = False
    print("\nAccuracy with t_max shifting: ")
    train_torch.evaluate_FC_SNN(model, dataset.test_load)
    print(f"output_layer.t_min={model.output_layer.t_min}\n\n")

    
    for i in range (20):
        data_label_tuple = dataset.test_set[i]
        x = data_label_tuple[0]
        y = model(x)
        y_predicted = torch.argmax(y)
        y_correct = data_label_tuple[1]
        print(f"y_pred={y_predicted} - y_corr={y_correct}")


    # config_utils.DEBUG_MODE = True
    # evaluate_FC_SNN(model, dataset.test_load)

    ''' Plot the membrane potential and spike times for selected neurons as 
        they were produced in the above forward pass. '''
    
    # output_activations = activations['layer_1']
    # sorted_index_activations = np.argsort(output_activations)
    # min_spike_neuron_index = sorted_index_activations[0]

    # plotting.plot_membrane_potential_path(model, dump_path, [min_spike_neuron_index, 40,200])
    # plotting.plot_output_spikes(dump_path, additional_title='\nSingle Forward Pass')


    ''' Attempt another forward pass on an MNIST image but also apply grayscale '''
    # model.collect_activations = True 
    # plotting.plot_input_tensor(x.view(28,28))

    # img = x.clone()

    # # Generate noise: values between -noise_level and 0
    # noise = -torch.rand_like(img) * 6.3  # Negative noise only

    # # Apply only to black pixels (value == 1.0)
    # black_mask = (img == 1.0)
    # img[black_mask] += noise[black_mask]

    # img.clamp(0.0, 1.0)
    # print(img)

    # plotting.plot_input_tensor(img.view(28,28))

    # y = model(img)
    # config_utils.logging.info(f"Model output: {y}")
    # path = args.logging_dir + 'outputs/' + args.model_name + '_gray.npz'
    # model.dump_activations(path)
    # plotting.plot_membrane_potential_path(model, path, [294, 40,200])
    # plotting.plot_output_spikes(path, additional_title='\nSingle Forward Pass - Gray Noise')


    ''' So far only the fully functional model has been tested. Now apply optimizations and experiments. '''
    # model.collect_activations = True 
    # train_torch.evaluate_FC_SNN(model, dataset.test_load)
    # unoptimized_activations = args.logging_dir + 'outputs/' + args.model_name + '_testing_unopt.npz'
    # model.dump_activations(unoptimized_activations)
    # model.collect_activations = False
    # plotting.plot_output_spikes(unoptimized_activations, log_scale=True, additional_title='\nNo optimizations')

    # ''' Apply optimizations '''
    # config_utils.logging.info("\n\n### Apply optimizations to model ###")

    # # Apply latency quantiles aat 99%, ..., 95% - evaluate and save activations
    # for q in reversed(range(95,100)):
    #     config_utils.logging.info(f"### Apply latency quantile q={q}%")
    #     model.apply_max_quantiles(q)

    #     # train_torch.evaluate_FC_SNN(model, dataset.test_load)
    #     y = model(x)
    #     optimization_name = f'_{q}.npz'
    #     optimized_activations_path = args.logging_dir + 'outputs/' + args.model_name + optimization_name
    #     model.dump_activations(optimized_activations_path)

    #     for n, layer in enumerate(model.hidden_layers):
    #         config_utils.logging.info(f"layer_{n}: t_max={layer.t_max}, t_max_q={layer.robustness_params['latency_quantiles'] * layer.t_max}")
    
    # # Plot activations after optimizations
    # for q in reversed(range(95,100)):
    #     optimization_name = f'_{q}.npz'
    #     optimized_activations_path = args.logging_dir + 'outputs/' + args.model_name + optimization_name
    #     plotting.plot_output_spikes(optimized_activations_path, log_scale=True, additional_title=f'\n{str(q)}')





''' -------------------- Membrane Potential Plots --------------------- '''
# if 'SNN' in args.model_type:
#     config_utils.logging.info("\nValidation Accuracy before adjusting latency quantiles: ")
#     train_torch.evaluate_FC_SNN(model, dataset.test_load)
#     # Get the index of the neuron that produced the very first spike in the next layer
#     output_activations = model.layer_activations[1]
#     sorted_output_activations = np.sort(model.layer_activations[1])
#     sorted_index_activations = np.argsort(model.layer_activations[1])
#     min_spike_neuron_index = sorted_index_activations[0]


#     config_utils.plot_input_spikes()


#     crop_quantile = model.hidden_layers[1].robustness_params['latency_quantiles']
#     print(model.layer_activations)
#     plotting.plot_membrane_potential(model, [min_spike_neuron_index, 140, 200], 
#         f"t_max_1={np.round(model.hidden_layers[1].t_max)} - crop={crop_quantile}")
    # config_utils.clean_spike_logs(model) 

    # print(model.layer_activations)

    # ''' ----------- Instantiate new SNNs with cropped latency quantiles ------------'''
    # config_utils.logging.info("\n### Adjusting latency quantiles ###")
    # for i in range(1,5):
    #     quantile = 1 - (i+1) * 0.01
    #     for layer in model.hidden_layers:
    #         layer.robustness_params["latency_quantiles"] = quantile
        
    #     config_utils.logging.info(f"--- Preserved Spikes at {quantile} quantile")
    #     evaluate_FC_SNN(model, dataset.test_load)  
    #     print("\n")
    
    # crop_quantile = model.hidden_layers[1].robustness_params['latency_quantiles']
    # title=f"t_max_1={np.round(model.hidden_layers[1].t_max,2)} - crop={crop_quantile}"

    # config_utils.DEBUG_MODE = True
    # print(model.layer_activations)
    # model.eval()
    # y = model(x)
    # plotting.plot_membrane_potential(model, [min_spike_neuron_index, 140, 200], title_addition=title) 
    # # print(model.layer_activations)
    # config_utils.DEBUG_MODE = False


    # ''' --------------  Threshold shifting ------------ '''

    # config_utils.logging.info("### Applying threshold adjustment ###")
    # for i, layer in enumerate(model.hidden_layers):
    #     min_layer_activation = np.min(model.layer_activations[i])

    #     shift = min_layer_activation - layer.t_min 
    #     t_max_new = layer.t_max - shift
    #     config_utils.logging.info(f'### Layer_{i} -- current t_min={layer.t_min}, t_max={layer.t_max}')
    #     config_utils.logging.info(f"### min_spike_time: {min_layer_activation} -- shift={shift} -- t_max_new = {t_max_new}")

    #     layer.t_max = t_max_new 
    #     if i == (len(model.hidden_layers)-1):   # last hidden layer
    #         model.output_layer.t_min = t_max_new
    #     else:
    #         model.hidden_layers[i+1].t_min = t_max_new
    
    # config_utils.logging.info(f"--- Evaluation after update --- ")
    # train_torch.evaluate_FC_SNN(model, dataset.test_load) 


    # config_utils.DEBUG_MODE = True
    # config_utils.clean_spike_logs(model)
    # model.eval()
    # y = model(x)
    # print(model.layer_activations)
    # title=f"t_max_1={np.round(model.hidden_layers[1].t_max,2)} - crop={crop_quantile} - adjusted threshold"
    # plotting.plot_membrane_potential(model, [min_spike_neuron_index, 140, 200], title_addition=title)
    # config_utils.clean_spike_logs(model) 
    # print(model.layer_activations)
    # config_utils.DEBUG_MODE = False