from dataset_torch import Dataset_Torch
from torchvision import datasets, transforms
from train_torch import train_FC_SNN, evaluate_FC_SNN
import config_utils
from model_torch import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os 
from PIL import ImageFilter


data_name = 'MNIST'
model_name = 'FC_demo'
logging_dir = './logs/'

TRAINING = False 

''' ----------------------------- Setup general parameters  ------------------------ '''
model_name = data_name + '-' + model_name
config_utils.set_up_logging(logging_dir, model_name)
robustness_params={
    'noise': 0.0,
    'time_bits': 0,
    'weight_bits': 0,
    'w_min': 1.0,
    'w_max': 1.0,
    'latency_quantiles': 0.0
}
dataset = Dataset_Torch(
    'MNIST',
    flatten=True,
    convert_ttfs = True,   
    ttfs_noise= 0.0,
)

# Create model instance and setup interval parameters  ------------------
model = create_torch_fc_model_SNN(layers=5, robustness_params=robustness_params)
config_utils.logging.info("### Setting SNN intervals ###")
boundaries = model.set_snn_intervals(0,1)


''' -------------------- Plot the input image, its tensor transformation and distribution --------------  '''
# Retrieve the original unprocessed input image from the MNIST dataset
original_train_set = datasets.MNIST(root='./datasets/MNIST', train=True, download=False)
first_image = original_train_set[0][0]

# first_image = first_image.filter(ImageFilter.GaussianBlur(3))

# Retrieve the pre-processed image with the custom transforms applied
transformed_input_tensor = dataset.train_set[0][0]  # this will have a shape of (784,) => reshape for display
reshaped_input_tensor = transformed_input_tensor.view(28, 28)

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Prepare axes for first plot
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

# Plot original unprocessed image   # TODO: potentially add a heatmap annotation here too
ax0.imshow(first_image, cmap='gray')
ax0.set_title("Original MNIST (28x28) Input Image")
ax0.axis("off")

# Plot the transformed input tensor
hmap = ax1.imshow(reshaped_input_tensor, cmap='gray_r')
colorbar = fig.colorbar(hmap, ax=ax1, fraction=0.046, pad=0.04, cmap="gray_r")
colorbar.set_label("Tensor Item Value")
ax1.set_title("Transformed Input Tensor - (28x28) heatmap")
plt.show()

# Plot the distribution of the input tensor
fig = plt.figure(figsize=(10, 5))

plt.hist(transformed_input_tensor.numpy(), bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.title("Distribution of the Input Tensor Values - tensor shape=(784, )")
plt.tight_layout()
plt.show()

''' --------------------- Plot the SNN intervals to visualize the behavior of the model  ------------------------- '''
# Using matplotlib.pyplot.barh() to visualize time intervals; 
# Code taken from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html (accessed 13/03)
fig, ax = plt.subplots()
y_pos = np.arange(model.N_layers-1)       # (-1) to skip the non-spiking output layer

for n, layer in enumerate(model.hidden_layers):

    # Add the integration time window of layer N
    width_integrate = layer.t_min - layer.t_min_prev
    start_integrate = layer.t_min_prev 
    ax.barh(y=n, width=width_integrate, left=start_integrate, color = 'orange', height =0.2, label = 'Integrate')

    # Add the spiking time window of layer N
    width_spiking = layer.t_max - layer.t_min 
    start_spiking = layer.t_min 
    ax.barh(y=n, width=width_spiking, left=start_spiking, color='green', height=0.2, label = 'Fire')

    # Plot the vertical time boundary between layers N and N+1
    ax.axvline(x=layer.t_max, linestyle='--', label="layer t_max")

ax.set_xlabel("Time")
layer_labels = [f"layer_{i}" for i in y_pos]
ax.set_yticks(y_pos, labels=layer_labels)
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
ax.set_title("Spiking Time Windows per hidden layer")
plt.show()

''' -------------------------  Training of the SNN network --------------------------- '''

save_path = logging_dir + 'weights_demo_SNN.pth'
if TRAINING: 
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005, weight_decay=1e-4)    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      
    loss_fn = nn.CrossEntropyLoss()
    config_utils.logging.info("### Train the SNN on the training set ###")
    train_FC_SNN(model, dataset.train_load, epochs=5, optimizer=optimizer, scheduler=scheduler)
    config_utils.logging.info("### Done training the SNN - save weights ###\n\n")
    torch.save(model.state_dict(), save_path)
else:
    config_utils.logging.info("### Load the trained SNN weights ###")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    config_utils.logging.info("### Evaluate on testset ###")
    # evaluate_FC_SNN(model, dataset.test_load)

# activations = np.load(config_utils.LOGGING_DIR + "activations.npz")

''' ------------------ Visualize distribution of spiking timestamps per layer during a (single) forward pass ------------ '''
config_utils.logging.info("### Performing forward pass on the sample input ###")
config_utils.DEBUG_MODE = True          
config_utils.clean_spike_logs() 
model.eval()
y = model(transformed_input_tensor)
config_utils.DEBUG_MODE = False 
config_utils.logging.info(f"### Model predicted label: {torch.argmax(y)}\n")

''' ----------------------  Plotting ----------------------- '''
bin_edges = [
    np.arange(1499, 1500.99, 0.1), 
    np.arange(2999, 3000.99, 0.1), 
    np.arange(4499, 4500.99, 0.1),
    np.arange(5999, 6000.99, 0.1),
]

data = model.layer_activations
total_points = 0
for layer in model.layer_activations:
    total_points += layer.shape[0]
# data = [np.round(x, decimals=2) for x in model.layer_activations] 
fig, ax = plt.subplots(figsize = (9, 9))

for i, (sublist, edges) in enumerate(zip(data, bin_edges)):
    min, max = np.amin(sublist), np.amax(sublist)
    plt.hist(sublist, bins='auto', alpha=0.5, label=f'Sublist {i+1}', edgecolor='black', range=[min, max])

xticks_new = []
for n, layer in enumerate(model.hidden_layers):
    t_max = layer.t_max 
    xticks_new.append(t_max)

xtick_pos = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
xtick_lab = ['1000', '1500\nt_max_layer_1', '2000','2500', '3000\nt_max_layer_2', '3500', '4000', '4500\nt_max_layer_3', '5000', '5500','6000\nt_max_layer_4']

plt.xticks(ticks=xtick_pos, labels=xtick_lab)
# plt.hist(data,bins=bin_edges, alpha=0.5)
# fig, ax = plt.subplots(figsize = (9, 9))
# ax.hist(flattened_data, bins=10, edgecolor="black")
plt.title(f"Distribution of activations per layer - single forward pass - N=340x4={total_points}")
plt.xlabel("Activation Values")
plt.ylabel("Frequency")
plt.show()

for x in data:
    print(x)
    print("\n\n")

'''
for n, layer in enumerate(model.hidden_layers):
    print(len(model.layer_activations[n]))
    print(model.layer_activations[n])
'''

'''  --------------------------------    Attempt new histogram plots  ------------------------------- '''
'''
num_bins = 20
bin_edges = [

    np.arange(1499, 1500.99, 0.1), 
    np.arange(2999, 3000.99, 0.1), 
    np.arange(4499, 4500.99, 0.1),
    np.arange(5999, 6000.99, 0.1),
]
bins = np.concatenate(bin_edges)
'''
bin_edges = np.array([
    0, 1499, 1499.1, 1499.2, 1499.3, 1499.4, 1499.5, 1499.6, 1499.7, 1499.8, 1499.9, 1500, 1500.1, 1500.2, 1500.3, 1500.4, 1500.5, 1500.6, 1500.7, 1500.8, 1500.9, 1501, 
    1502,
    2998, 2999, 2999.1, 2999.2, 2999.3, 2999.4, 2999.5, 2999.6, 2999.7, 2999.8, 2999.9,
    3000.0, 3000.1, 3000.2, 3000.3, 3000.4, 3000.5, 3000.6, 3000.7, 3000.8, 3000.9, 3001.0,
    3002,
    4499.0, 4499.1, 4499.2, 4499.3, 4499.4, 4499.5, 4499.6, 4499.7, 4499.8, 4499.9, 4500.0, 
    4500.1, 4500.2, 4500.3, 4500.4, 4500.5, 4500.6, 4500.7, 4500.8, 4500.9, 4501.0,
    4502, 
    5997, 5998,
    5998.1, 5998.2, 5998.3, 5998.4, 5998.5, 5998.6, 5998.7, 5998.8, 5998.9, 5999.0, 5999.1, 5999.2, 5999.3, 5999.4, 5999.5, 5999.6, 5999.7, 5999.8, 5999.9, 
    6000.0, 6000.1, 6000.2, 6000.3, 6000.4, 6000.5, 6000.6, 6000.7, 6000.8, 6000.9, 6001.0,
    6002      
])
plt.hist(data, bins=bin_edges, alpha=0.7, edgecolor='black')
plt.ylim(0,50)
plt.title('Distribution of activations per layer - single forward pass - N=340x4={total_points} - custom bin edges')
plt.xlabel("Activation Values")
plt.ylabel("Frequency")
plt.show()

'''-------------------------------------------------------------------------------------------------------'''
# Read the input/output spike times from the logs
path = config_utils.LOGGING_DIR + 'spike_output.txt'
total_spikes = 0
spikes_per_layer = []
spikes_per_layer.append(transformed_input_tensor.tolist())
with open(path, 'r') as f:
    for line in f:
        line = line.strip("\n")  
        spikes = [round(float(v), 2) for v in line.split()]
        spikes_per_layer.append(spikes)
        total_spikes += len(spikes)

labels = ["Input Layer"] + layer_labels
labels = layer_labels
plt.hist(x = spikes_per_layer, label=labels, bins=100, alpha=0.7, edgecolor="black", rwidth=0.7)      
plt.xlim(0, None)
plt.xlabel("Layer Output // Spike Time")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings per layer - single forward pass on Input(x) - N={(total_spikes)} data points")
plt.grid(axis="y", linestyle="--", alpha=0.7) 

for n, layer in enumerate(model.hidden_layers):
    t_max = layer.t_max 
    plt.axvline(x=t_max, color='red', label="Layer t_max", linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
plt.show()

''' -----------------------------------------------------------------------------------'''


fig = plt.figure(figsize=(12, 5))
bins = 50
plt.hist(model.list_activations[0], bins=bins, alpha=0.7)
plt.axvline(x=1501.0, color='red', label="Layer[0] t_max", linestyle='--')
print(model.list_activations[0])
plt.xlabel("Activation Values")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings for layer[0] - forward pass on input - N={len(model.layer_activations[0])} bins={bins}")
plt.legend()
plt.show()


''' ------------------ Visualize distribution of spiking timestamps per layer during entire training process ------------ '''
'''
config_utils.logging.info("### Visualize spiking time activations during training phase ###")
fig = plt.figure(figsize=(12, 5))

total_points = 0
bins = 100


for layer_name in activations.files:
    plt.hist(activations[layer_name].flatten(), bins=50, alpha=0.7)


for i in range(len((model.layer_activations))):
    config_utils.logging.info(f"### Layer {i}: {len(model.layer_activations[i])} layer output activations // spike timing timestamps")
    plt.hist(model.layer_activations[i], bins=bins, alpha=0.7, log=True)
    total_points += len(model.layer_activations[i])

print("-----------------")
print(type(model.layer_activations[0]))'


# plt.hist(model.layer_activations[0], bins=100, alpha=0.7, log=True)

# plt.xlim(0, None)
# plt.yscale("log")
plt.xlabel("Output Activations // Spike Timing Timestamps")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings per layer - training phase, all layers - N={total_points} - bins={bins}")
plt.grid(axis="y", linestyle='--', alpha=0.6)
plt.show()
config_utils.logging.info("###\n")
# ------------------------------------------------------------------------------------------------------------------------------------- # 
fig = plt.figure(figsize=(12, 5))

total_points = 0
bins = 20

for layer_name in activations.files:
    plt.hist(activations[layer_name].flatten(), bins=50, alpha=0.7)



for i in range(len((model.layer_activations))):
    config_utils.logging.info(f"### Layer {i}: {len(model.layer_activations[i])} layer output activations // spike timing timestamps")
    plt.hist(model.layer_activations[i], bins=bins, alpha=0.9, log=True, density=True)
    total_points += len(model.layer_activations[i])

plt.xlim(0, None)
# plt.yscale("log")
plt.xlabel("Output Activations // Spike Timing Timestamps")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings per layer - training phase, all layers - N={total_points} - bins={bins}")
plt.grid(axis="y", linestyle='--', alpha=0.6)
plt.show()
config_utils.logging.info("###\n")
# -------------------------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 5))

total_points = 0
bins = 20

for layer_name in activations.files:
    data = np.round(activations[layer_name].flatten(), decimals=2)
    plt.hist(activations[layer_name].flatten(), bins=50, alpha=0.7)


'''
'''
for i in range(len((model.layer_activations))):
    config_utils.logging.info(f"### Layer {i}: {len(model.layer_activations[i])} layer output activations // spike timing timestamps")
    data = np.round(model.layer_activations[i], decimals=2)
    plt.hist(data, bins=bins, alpha=0.9, log=True, density=True)
    total_points += len(model.layer_activations[i])

plt.xlim(0, None)
# plt.yscale("log")
plt.xlabel("Output Activations // Spike Timing Timestamps")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings per layer - training phase, all layers - N={total_points} - bins={bins} - np.round=2")
plt.grid(axis="y", linestyle='--', alpha=0.6)
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 5))
bins = 20
plt.hist(model.layer_activations[0], bins=bins, alpha=0.7, density=True)

plt.xlim(0, None)
plt.yscale("log")
plt.xlabel("Output Activations // Spike Timing Timestamps")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings for layer[0] - training phase - N={len(model.layer_activations[0])} bins={bins}")
plt.show()
# ---------------------------------------------------------------------------- #

fig = plt.figure(figsize=(12, 5))
bins = 50
plt.hist(model.layer_activations[0], bins=bins, alpha=0.7, density=True)
plt.xlabel("Output Activations // Spike Timing Timestamps")
plt.ylabel("Frequency")
plt.title(f"Distribution of Spike Timings for layer[0] - training phase - bins={bins} - cropped to layer's t_max range")
plt.show()

data = np.round(model.layer_activations[0], decimals=2)
unique, counts = np.unique(data, return_counts=True)
print(np.asarray((unique, counts)).T)


# --------------------------------------------------------------------------------  #
#-----------------------------------------------------------------------------------#
config_utils.logging.info("### Evaluation on testset ###")
evaluate_FC_SNN(model, dataset.test_load)'
'''