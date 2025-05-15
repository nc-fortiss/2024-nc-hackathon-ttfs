import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # reduce TF verbosity
os.environ['CUDA_VISIBLE_DEVICES']='0'  #'-1' for CPU only
import argparse
import pickle as pkl
from Dataset import Dataset
from model import *
import time
start_time = time.time()
tf.keras.backend.set_floatx('float64') #to avoid numerical differences when comparing training of ReLU vs SNN
override = None
import h5py
import matplotlib.pyplot as plt
import utils

# example run: python3 main.py --data_name=MNIST --model_type=SNN --model_name=FC2 --testing=False --epochs=1
# hint for debugging: to print the values of a tensor, use tf.get_static_value(tensor_input) or also tf.keras.backend.eval(y_all)
# import pdb


strtobool = (lambda s: s=='True')
parser = argparse.ArgumentParser(description='TTFS')
parser.add_argument('--data_name', type=str, default='MNIST', help='(MNIST|CIFAR10|CIFAR100)')
parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')
parser.add_argument('--model_type', type=str, default='SNN', help='(SNN|ReLU)')
parser.add_argument('--model_name', type=str, default='FC2', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Epochs. 0 -skip training')
parser.add_argument('--layers', type=int, default=2, help='Number of layers in FC model')
parser.add_argument('--testing', type=strtobool, default=False, help='Execute testing.')
parser.add_argument('--load', type=str, default='False', help='Load before training. (True|False|custom_name.h5)')
parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
parser.add_argument('--train_shift', type=strtobool, default=True, help='Re-calculate interval boundaries during training.')
# Robustness parameters:
parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument('--mode', type=str, default='', help='Ignore: A hack to address a bug in argsparse during debugging')
args = parser.parse_known_args(override)
if(len(args[1])>0):
    print("Warning: Ignored args", args[1])
args = args[0]
args.model_name = args.data_name + '-' + args.model_name
utils.TRAIN_SHIFT = args.train_shift
utils.LOGGING_DIR = args.logging_dir
set_up_logging(args.logging_dir, args.model_name)
robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'w_min': args.w_min,
    'w_max': args.w_max,
    'latency_quantiles':args.latency_quantiles
}

# Create data object
data = Dataset(
    args.data_name,
    args.logging_dir,
    flatten='FC' in args.model_name,
    ttfs_convert='SNN' in args.model_type,
    ttfs_noise=args.noise,
)
# Get optimizer for training.
optimizer = get_optimizer(args.lr)
model = None
logging.info("#### Creating the model ####")
if 'FC2' in args.model_name:
    if 'SNN' in args.model_type:
        model = create_fc_model_SNN(layers=args.layers,X_n=[10,50] , optimizer=optimizer, robustness_params=robustness_params)
    if 'ReLU' in args.model_type:
        model = create_fc_model_ReLU(layers=args.layers, optimizer=optimizer)
if 'VGG' in args.model_name:
    # We consider one architecture, a 15-layer VGG-like network.
    if 'MNIST' in args.data_name: #MNIST / FMNIST
        layers2D=[64, 64,       128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D=[512, 512]
    else:  #other: CIFAR10, CIFAR100
        layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D=[512]
    kernel_size=(3,3)
    regularizer = None
    initializer = 'glorot_uniform' # keras default
    BN = 'BN' in args.model_name
    if not BN:  # resort to settings similar to the initial VGG paper
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
        regularizer = tf.keras.regularizers.L2(5e-4)
        initializer = 'he_uniform'
    if 'SNN' in args.model_type:
        model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params=robustness_params,
                                     kernel_regularizer=regularizer, kernel_initializer=initializer)
    if 'ReLU' in args.model_type:
        model = create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=BN, optimizer=optimizer,
                                       kernel_regularizer=regularizer, kernel_initializer=initializer)
if model is None:
    print('Please specify a valid model. Exiting.')
    exit(1)
model.summary()
model.last_dense = list(filter(lambda x : 'dense' in x.name, model.layers))[-1]




if args.load != 'False':
    logging.info("#### Loading weights ####")
    if 'ReLU' in args.model_type:
        # Load weights
        if args.load == 'True':  # automatic name
            model.load_weights(args.logging_dir + args.model_name + '_weights.h5', by_name=True)
        else:  # custom name
            model.load_weights(args.logging_dir + args.load, by_name=True)
    if 'SNN' in args.model_type:
        # Load X_n ranges from the max ANN activations, if available 
        if os.path.exists(args.logging_dir + args.model_name + '_X_n.pkl'):
            logging.info("### Found X_n ranges from ANN conversion")
            X_n=pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))
        else:
            X_n = [10,50]
        if 'FC2' in args.model_name:
            logging.info(f"### Create new FC instance with loaded X_n = {X_n} ###\n")
            model = create_fc_model_SNN(layers=args.layers, optimizer=optimizer, X_n=X_n, robustness_params=robustness_params)
        else:
            logging.info(f"### Create new VGG instance with loaded X_n = {X_n} ###\n")
            model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, X_n=X_n, robustness_params=robustness_params,
                                     kernel_regularizer=regularizer, kernel_initializer=initializer)
        
        def explore_group(g, indent=0):
            for key in g:
                item = g[key]
                if isinstance(item, h5py.Group):
                    print("  " * indent + f"[Group] {key}")
                    explore_group(item, indent + 1)
                elif isinstance(item, h5py.Dataset):
                    print("  " * indent + f"{key}: shape={item.shape}")
                else:
                    print("  " * indent + f"{key}: unknown type {type(item)}")

        # check if a fully-trained SNN model exists
        if os.path.exists(args.logging_dir + '/' + args.model_name + '_full_SNN_weights.h5'):

            # Open the HDF5 file
            with h5py.File(args.logging_dir + '/' + args.model_name + '_full_SNN_weights.h5', 'r') as f:
                explore_group(f)


            # model.load_weights(args.logging_dir + '/' + args.model_name + '_full_SNN_weights.h5', by_name=True)
        # check if a convertible ANN model exists
        elif os.path.exists(args.logging_dir + args.model_name + '_preprocessed.h5'):
            model.load_weights(args.logging_dir + args.model_name + '_preprocessed.h5', by_name=True)
        else: 
            logging.info("### !! There are no pre-trained weights to load for the SNN !!")
            exit(1)



if 'SNN' in args.model_type:
    # Register SNN interval boundaries at initialization before training
    logging.info("#### Set SNN intervals BEFORE training ####")
    t_min, t_max = 0, 1 
    for n, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            t_min, t_max = layer.set_params(t_min, t_max)
            logging.info(f"layer_{n}: set t_min={layer.t_min}, t_max={layer.t_max}, B_n={layer.B_n}")
    logging.info("\n\n")
   
# if args.testing != False:
#     logging.info("#### Initial test set accuracy testing ####")
#     test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
#     logging.info("Initial testing accuracy is {}.".format(test_acc))

### Plot input coding
image_index = 15

fig, ax = plt.subplots(1, 3, figsize=(12, 8))
(x_train,y_train), (x_test,y_test)=tf.keras.datasets.cifar10.load_data()
x_original = x_train[image_index]

### Plot original image
ax[0].imshow(x_original)
ax[0].set_title(f"Original CIFAR10 Image - {x_original.shape}")
ax[0].axis('off')

### Plot normalized image
x_norm = (x_original-120.707)/(64.15+1e-7)
ax[1].imshow(x_norm)
ax[1].set_title(f"Normalized CIFAR10 Image - {x_norm.shape}")
ax[1].axis('off')

### Plot TTFS conversion
x = data.x_train[image_index]
ax[2].imshow(x, cmap='gray_r')
ax[2].set_title(f"TTFS CIFAR10 Image - {x_norm.shape}")

plt.tight_layout()
plt.show()


logging.info("#### Attempt a single forward pass ####")
x_expanded = tf.expand_dims(x, axis=0)
model(x_expanded)



# fig, ax = plt.subplots(3, 2, figsize=(10, 8))  # Now 3 rows × 2 cols
# plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust horizontal spacing

# ax[0, 0].set_title("TTFS Input", pad=50)
# ax[0, 1].set_title("Input Spike Distribution", pad=50)

# for i in range(3):
#     channel_np = x[:,:,i]
    
#     # --- Left Column: Image + Text Label ---
#     im = ax[i, 0].imshow(channel_np, cmap='gray_r')
    
#     # Add channel label inside the image subplot (left-aligned, vertically centered)
#     ax[i, 0].text(-0.65, 0.5, f'Channel {i+1}', 
#                  transform=ax[i, 0].transAxes,  # Uses axes coordinates (0-1)
#                  ha='right', va='center',
#                  fontsize=12,
#                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

#     # cbar = fig.colorbar(im, ax=ax[i, 0], fraction=0.046, pad=0.05, location='left')
#     # cbar.set_label('Intensity', fontsize=10) 
    
#     # --- Right Column: Histogram ---
#     ax[i, 1].hist(channel_np.flatten(), bins=20, alpha=0.4)
#     ax[i, 1].set_xlim(0, 1)
#     ax[i, 1].set_xlabel("Spike Time")  
#     ax[i, 1].set_ylabel("Frequency") 

# plt.show()

# spike_times, t_max_values = [], []
# model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs[0])
# layer_names = []
# for k, layer in enumerate(model.layers):

#     if 'conv2d' in layer.name or 'dense' in layer.name:
#         if k!=len(model.layers)-2:
#             spike_times.append(layer.output)
#             t_max_values.append(layer.t_max)
#             layer_names.append(layer.name)
        
# model.compile(metrics=["categorical_accuracy"], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                 optimizer=optimizer)    


# extractor_spks = tf.keras.Model(inputs=model.inputs, outputs=spike_times)
# output_intermediate_spikes = extractor_spks.predict(x_expanded, verbose=1)

# x_expanded = tf.expand_dims(x, axis=0)
# model(x_expanded)

# plt.figure(figsize=(10,6))
# # Plot latency distribution
# if output_intermediate_spikes is None:
#     print("there is nothing to plottttt")
# for i in range(len(output_intermediate_spikes)):
#     t_max_layer = t_max_values[i].numpy()
#     output_flat = output_intermediate_spikes[i].flatten()
#     # output_flat = output_flat[output_flat < t_max_layer]
#     # Plot the combined histogram
#     if i == 13: break
#     output_shape = output_flat.shape[0]
#     plt.hist(output_flat, bins=10, density=True, label=f'{layer_names[i]} - N={output_shape}')

#     print(f"--- {layer.name} ---")
#     print(f"--- mean={np.mean(output_flat)}; min={np.min(output_flat)}; max={np.max(output_flat)}")

# plt.title('Layer-wise Activations - CIFAR10 VGG16 Inference')
# plt.xlabel('Spiking Time')
# plt.ylabel('Frequency')
# plt.legend(loc='upper left')
# plt.grid(True)
# plt.show()




# breakpoint()

if args.epochs > 0:
    logging.info("#### Training ####")
    history=model.fit(
        data.x_train, data.y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(data.x_test, data.y_test)
        )

if args.testing and args.epochs > 0:
    # Obtain accuracy of the fine-tuned SNN model.
    logging.info("#### Final test set accuracy testing ####")
    test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
    logging.info("Final testing accuracy is {}.".format(test_acc))

if args.save and 'ReLU' in args.model_type:
    logging.info("\n\n#### Saving ReLU model ####")
    # breakpoint()
    # 1. Save original ReLU weights
    model.save_weights(args.logging_dir + '/' + args.model_name + '_weights.h5')

    # Fuse (imaginary) batch normalization layers.
    logging.info('fuse (imaginary) BN layers')
    # shift/scale input data accordingly


    data.x_test, data.x_train = (data.x_test - data.p)/(data.q-data.p), (data.x_train - data.p)/(data.q-data.p)
    BN = 'BN' in args.model_name 
    model = fuse_bn(model, BN=BN, p=data.p, q=data.q, optimizer=optimizer)
    logging.info(model.summary())

    # 2. Save preprocessed ReLU model.
    model.save_weights(args.logging_dir + '/' + args.model_name + '_preprocessed.h5')
    logging.info('saved preprocessed ReLU model')

    logging.info(f"### x_train={type(data.x_train)}")

    # 3. Find maximum layer outputs.
    logging.info('calculating maximum layer output...')
    layer_num, X_n = 0, []
    layers_max = []
    layers_min = []     # also track the minimum spike times to crop the intervals later in the SNN
    for k, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            if k!=len(model.layers)-2:
                # Calculate X_n of the current layer.
                layers_max.append(tf.reduce_max(tf.nn.relu(layer.output)))
                # logging.info(f"k={k} --- layers_max={layers_max}")
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layers_max)
    print(extractor.summary())
    output = extractor.predict(data.x_train, batch_size=64, verbose=1)
    X_n = list(map(lambda x: np.max(x), output))
    logging.info('X_n: %s', X_n)
    pkl.dump(X_n, open(args.logging_dir + '/' + args.model_name + '_X_n.pkl', 'wb'))
    logging.info('saved maximum layer output')

if args.save and 'SNN' in args.model_type:
    logging.info("\n\n#### Saving fully trained SNN model weights ###")
    model.save_weights(args.logging_dir + '/' + args.model_name + '_full_SNN_weights.h5')

print('### Total elapsed time [s]:', time.time() - start_time)
print("\n")
logging.info("#### Attempt a single forward pass ####")
x = data.x_train[0]
x_expanded = tf.expand_dims(x, axis=0)
y = model(x_expanded)

if 'SNN' in args.model_type:


    ''' Apply layer-wise threshold adjustment '''
    logging.info("####\n\n Apply threshold adjustment ####")

    # Check if the min_spikes have already been extracted before - if yes, load those
    if os.path.exists(args.logging_dir + '/' + args.model_name + '_min_spikes.pkl'):
        global_minima = pkl.load(open(args.logging_dir + '/' + args.model_name + '_min_spikes.pkl', 'rb'))
    else: 
        # Extract the min_spikes and save them
        # First, make a pass over the testset: extract the global minimum spike time per layer
        min_spike_times = []
        for k, layer in enumerate(model.layers):
            if 'conv' in layer.name or 'dense' in layer.name:
                min_spike_times.append(tf.reduce_min(layer.output))

        extractor = tf.keras.Model(inputs=model.inputs, outputs=min_spike_times)
        min_layer_outputs = extractor.predict(data.x_test, batch_size=8, verbose=1)
        
        # Global minimum spike times
        global_minima = list(map(lambda x: np.max(x), min_layer_outputs))
        print(f"--- Extracted min_spike times: {global_minima} ---")
        pkl.dump(global_minima, open(args.logging_dir + '/' + args.model_name + '_min_spikes.pkl', 'wb'))

    # Apply threshold modificationa
    t_max_new = 1
    k = 0
    for layer in model.layers:
        if 'conv' in layer.name or 'dense' in layer.name:
            logging.info(f"--- prev: t_min={layer.t_min}  - t_max={layer.t_max}")
            layer.t_min.assign(t_max_new)
            t_max_new = layer.t_max + layer.t_min - global_minima[k]
            layer.t_max.assign(t_max_new)
            logging.info(f"--- new : t_min={layer.t_min} - t_max={layer.t_max}")
            k+=1
    
    # if args.testing != False:
    #     # Re-evaluate accuracy
    #     logging.info("--- Accuracy after threshold adjustment: ")
    #     test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
    #     logging.info("--- Adjusted threshold - testing accuracy is {} ---".format(test_acc))


    # Re-plot adjusted activations for a single pass
    spike_times_adjusted = []
    for k, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            spike_times_adjusted.append(layer.output)

    extractor = tf.keras.Model(inputs=model.inputs, outputs=spike_times_adjusted)
    layer_outputs_adjusted = extractor.predict(x_expanded, verbose=1)

    for i in range(len(layer_outputs_adjusted)):
        output_flat = layer_outputs_adjusted[i].flatten()  
        # output_filter = output_flat[output_flat < t_max_layer] 
        output_filter = output_flat
        plt.hist(output_filter, bins=100, alpha=0.5)

    t_max_layer_before = t_max_layer
    plt.title('Layer-wise Adjusted Activations - CIFAR10 VGG16 Inference')
    plt.xlabel('Spiking Time')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


    # ------------------------------------------------------------------------------------------------------------------ # 
    # ### Apply 2nd optimization: interval overlapping by shifting t_max
    logging.info("\n\n Apply interval overlapping optimization")
    new_t_max = 1
    for layer in model.layers:
        if 'conv' in layer.name or 'dense' in layer.name:
            logging.info(f"--prev t_min={layer.t_min} - t_max={layer.t_max}")
            layer.t_min = new_t_max
            new_t_max = 0.5 * layer.t_max 
            layer.t_max = new_t_max
            logging.info(f"-- new t_min={layer.t_min} - t_max={layer.t_max}")
        

    # Evaluate testset with overlap
    if args.testing != False:
        logging.info("--- Accuracy after interval overlap adjustment: ")
        test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
        logging.info("--- Overlapping Intervals - testing accuracy is {} ---".format(test_acc))

    # Make a forward pass
    spike_times_overlap = []
    for k, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            spike_times_overlap.append(layer.output)

    extractor = tf.keras.Model(inputs=model.inputs, outputs=spike_times_overlap)
    layer_outputs_overlap = extractor.predict(x_expanded, verbose=1)

    for i in range(len(layer_outputs_overlap)):
        output_flat = layer_outputs_overlap[i].flatten()  
        # output_filter = output_flat[output_flat < t_max_layer] 
        output_filter = output_flat
        plt.hist(output_filter, bins=100, alpha=0.5)

    plt.title('Layer-wise Overlapping Activations - CIFAR10 VGG16 Inference')
    plt.xlabel('Spiking Time')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


logging.info(y)
logging.info(data.y_train[0])



if 'SNN' in args.model_type:
    logging.info("### Printing layer intervals AFTER training ###")
    for n, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            logging.info(f"layer_{n}: t_min={layer.t_min}, t_max={layer.t_max}, B_n={layer.B_n}")
    logging.info("\n\n")
