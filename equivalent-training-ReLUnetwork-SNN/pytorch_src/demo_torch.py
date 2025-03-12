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

''' Plotting of the input data and its transforms '''

# Retrieve the original unprocessed input image from the MNIST dataset
original_train_set = datasets.MNIST(root='./datasets/MNIST', train=True, download=False)
first_image = original_train_set[0][0]

# Retrieve the pre-processed image with the custom transforms applied
transformed_input_tensor = dataset.train_set[0][0]  # this will have a shape of (784,) => reshape for display
reshaped_input_tensor = transformed_input_tensor.view(28, 28)

# Start plot for input
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Prepare axes for first plot
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

# Plot original unprocessed image
# TODO: potentially add a heatmap annotation here too
ax0.imshow(first_image, cmap='gray')
ax0.set_title("Original MNIST (28x28) Input Image")
ax0.axis("off")

# Plot the transformed input tensor
hmap = ax1.imshow(reshaped_input_tensor, cmap='gray')
colorbar = fig.colorbar(hmap, ax=ax1, fraction=0.046, pad=0.04)
colorbar.set_label("Item Value")
ax1.set_title("Transformed Input Tensor - (28x28) heatmap")

plt.show()

# Plot the distribution of the input tensor
fig = plt.figure(figsize=(10, 5), constrained_layout=True)

plt.hist(transformed_input_tensor.numpy(), bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.title("Distribution of the Input Tensor Values - tensor shape=(784, )")
plt.show()

''' Training of the SNN network '''

# Create model instance and setup interval parameters
model = create_torch_fc_model_SNN(layers=3, robustness_params=robustness_params)
model.set_snn_intervals(0,1)

# Train the model
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005, weight_decay=1e-4)    # applying regularization (weight_decay) turns out to be crucial for training

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)      # step-wise learning rate adjustment
loss_fn = nn.CrossEntropyLoss()

config_utils.logging.info("### Train the SNN on the training set ###")
train_FC_SNN(model, dataset.train_load, epochs=1, optimizer=optimizer, scheduler=scheduler)

config_utils.logging.info("### Evaluation on testset ###")
evaluate_FC_SNN(model, dataset.test_load)

# Pass the input image from above to test a forward pass 
config_utils.DEBUG_MODE = True          
config_utils.clean_spike_logs() 
model.eval()
y = model(transformed_input_tensor)

config_utils.logging.info(f"### Model predicted label: {torch.argmax(y)}\n")


# Read the input/output spike times from the logs
path = config_utils.LOGGING_DIR + 'spike_output.txt'
spikes_per_layer = []
with open(path, 'r') as f:
    for line in f:
        line = line.strip("\n")  
        spikes = [float(v) for v in line.split()]  
        spikes_per_layer.append(spikes)

plt.hist(spikes_per_layer, bins=20, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Fine-Grained Histogram (Flattened 2D List)")
plt.grid(axis="y", linestyle="--", alpha=0.7) 

plt.show()
