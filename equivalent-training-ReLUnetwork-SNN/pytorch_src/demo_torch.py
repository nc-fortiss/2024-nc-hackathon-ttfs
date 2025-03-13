from dataset_torch import Dataset_Torch
from torchvision import datasets, transforms
from train_torch import train_FC_SNN, evaluate_FC_SNN
import config_utils
from model_torch import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


data_name = 'MNIST'
model_name = 'FC_demo'
logging_dir = './logs/'

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
hmap = ax1.imshow(reshaped_input_tensor)
colorbar = fig.colorbar(hmap, ax=ax1, fraction=0.046, pad=0.04)
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
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005, weight_decay=1e-4)    
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      
loss_fn = nn.CrossEntropyLoss()
config_utils.logging.info("### Train the SNN on the training set ###")
train_FC_SNN(model, dataset.train_load, epochs=1, optimizer=optimizer, scheduler=scheduler)

config_utils.logging.info("### Evaluation on testset ###")
evaluate_FC_SNN(model, dataset.test_load)

# Pass the input image from above to test a forward pass 
config_utils.logging.info("### Perform forward pass on the sample input ###\n")
config_utils.DEBUG_MODE = True          
config_utils.clean_spike_logs() 
model.eval()
y = model(transformed_input_tensor)
config_utils.logging.info(f"### Model predicted label: {torch.argmax(y)}\n")

''' ------------------ Visualize distribution of spiking timestamps per layer during a forward pass ------------ '''
# Read the input/output spike times from the logs
path = config_utils.LOGGING_DIR + 'spike_output.txt'
spikes_per_layer = []
spikes_per_layer.append(transformed_input_tensor.tolist())
with open(path, 'r') as f:
    for line in f:
        line = line.strip("\n")  
        spikes = [round(float(v), 2) for v in line.split()]
        spikes_per_layer.append(spikes)
# print(spikes_per_layer)


labels = ["Input Layer"] + layer_labels
labels = layer_labels
plt.hist(x = spikes_per_layer, label=labels, bins=100, alpha=0.7, edgecolor="black", rwidth=0.7)      
plt.xlim(0, None)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Distribution of Spike Timings per layer")
plt.grid(axis="y", linestyle="--", alpha=0.7) 

for n, layer in enumerate(model.hidden_layers):
    t_max = layer.t_max 
    plt.axvline(x=t_max, color='red', label="Layer t_max", linestyle='--')

handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
plt.show()


''' --------------- Visualize distribution of spiking timestamps across the entire training process------------ '''
