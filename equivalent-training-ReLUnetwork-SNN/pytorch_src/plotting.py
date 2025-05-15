
import numpy as np
from model_torch import compute_membrane_potential
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config_utils
import os 
from model_torch import *
from train_torch import * 
import dataset_torch
import pickle as pkl 
from torchvision import datasets, transforms

def plot_membrane_potential(model, target_neurons, title_addition=''):
    '''
        Plots the membrane potential dynamics for 'target_neurons'.

        Arguments:
        @model: FC_SNN_torch instance. The model should have at least 3 hidden layers (including the output_layer)
        @target_neurons: can be a list or an integer representing the target neuron
            for which the membrane potential(s) should be plotted
        @title_addition: additional string for extending the plot title
    '''
    # Convert integer into list if a list was not passed (simplifies later code)
    if not isinstance(target_neurons, list):
        target_neurons = [target_neurons]


    # Spike activations arriving from layer (N-1)
    input_activations = model.layer_activations[0]
    sorted_activations = np.sort(model.layer_activations[0])
    sorted_indices = np.argsort(input_activations)

    # New output spikes generated in layer (N)
    output_activations = model.layer_activations[1]
    sorted_output_activations = np.sort(model.layer_activations[1])

    # Get the spike threshold for the i-th neuron. In particular, 'D_i' is the trainable parameter that is unique to each neuron
    thresholds = {}
    for neuron in target_neurons:
        thresholds[neuron] = model.hidden_layers[1].t_max - model.hidden_layers[1].t_min - model.hidden_layers[1].D_i[neuron].detach().numpy()

    # For all neurons (i) in 'target_neurons':
    # Loop over a selection of neurons (j) from layer (N-1) which have sent a spike to neuron(i) in layer(N)
    # through the synapse (W_ji); collect spike times and membrane potentials for these neurons

    spike_times = {}
    potentials = {}
    for target_neuron_index in target_neurons:
        
        spike_times_i = []        # input spiking timestamps (tj) from layer (N-1) to neuron(i)
        potentials_i = []         # change in membrane potential neuron (i) in layer (N)
        for j in range(1,340,10):
            V_i = compute_membrane_potential(
                    sorted_activations[j],      # get next input spike time tj at which V should be evaluated
                    target_neuron_index,        
                    model.layer_activations[0],         # all input spikes tj from layer (N-1)  
                    model.hidden_layers[1].kernel.data.detach().numpy()     # kernel W connecting layer (N-1) to layer (N)
                )

            spike_times_i.append(sorted_activations[j])
            potentials_i.append(V_i)

        spike_times[target_neuron_index] = spike_times_i 
        potentials[target_neuron_index] = potentials_i
    
    # print("Spike Times: ", spike_times)
    # print("Potentials: ", potentials)
    # breakpoint()

    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14,            # general font size
        'axes.labelsize': 16,       # x/y label font
        'axes.titlesize': 18,       # title font
        'legend.fontsize': 13,      # legend font
        'xtick.labelsize': 12,      # tick font
        'ytick.labelsize': 12,
        'font.family': 'serif',     # or 'sans-serif', 'Times New Roman' for papers
    })


    # Start the plot 
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)

    # --- Interval boundaries (t_min, t_max) --- #
    ax.axvline(x=model.hidden_layers[0].t_min, linestyle="--", color="black")
    ax.axvline(x=model.hidden_layers[0].t_max, linestyle="--", color="black")
    ax.axvline(x=model.hidden_layers[1].t_max, linestyle="--", color="black")

    
    
    # --- Threshold of layer (N) --- #
    ax.axhline(y=thresholds[target_neurons[0]], label=f"threshold", color='green', lw=4, alpha=0.7)

    # --- Membrane potential changes --- #
    colors = cm.tab10(np.linspace(0, 1, len(target_neurons))) 
    colors = dict(zip(target_neurons, colors))
    for i in target_neurons:
        ax.scatter(spike_times[i], potentials[i], alpha=0.5, color=colors[i])
    
    # --- Constant slope - fire phase --- #
    for i in target_neurons:
        x_spike = thresholds[i] + (model.hidden_layers[0].t_max - potentials[i][-1])
        y_spike = x_spike - model.hidden_layers[0].t_max + potentials[i][-1]
        ax.plot([model.hidden_layers[0].t_max, x_spike], 
                [potentials[i][-1], y_spike], 
                color=colors[i], alpha=0.5, lw=4,
                label=f"V_{i}")
    
    y_min, y_max = ax.get_ylim()
    output_spike_times = []
    # --- Generated Spike Output Times in layer(N) --- #
    for i in target_neurons:
        # get the spike times that were produced by each neuron after integration and threshold crossing
        output_spike_from_neuron = model.layer_activations[1][i]
        output_spike_times.append(output_spike_from_neuron)
        ax.plot(output_spike_from_neuron, 0, color=colors[i], marker='^', markersize=15, label=f"Spike_{i}")
        ax.plot([output_spike_from_neuron, output_spike_from_neuron],
                [y_min, 0], linestyle="--", color=colors[i], alpha=0.5)


    # --- Axis labels --- #
    ax.set_xlabel('Spike Activation Time (t)', labelpad=20)
    ax.set_ylabel('Membrane Potential V(t)', labelpad=20)

    # Label the vertical lines for interval boundaries
    y_pos = (y_min) + (y_max - y_min) / 1.1     # position of (t_min) or (t_max) label
    ax.text(model.hidden_layers[0].t_min + 0.3, y_pos, 't_min_0', va='center', ha='left')
    ax.text(model.hidden_layers[0].t_max + 0.3, y_pos, 't_max_0\nt_min_1', va='center', ha='left')
    ax.text(model.hidden_layers[1].t_max + 0.3, y_pos, 't_max_1', va='center', ha='left')

    # xticks for generated spike times
    current_xticks = ax.get_xticks()
    new_xticks = sorted(set(current_xticks.tolist() + output_spike_times))
    ax.set_xticks(new_xticks)
    ax.set_xlim(0.0)

    
    title = 'Membrane Potential vs Spike Activations\nfor selected Neurons during Inference on a single MNIST image'
    if title_addition: title = title + "\n" + title_addition
    fig.suptitle(title)

    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))

    plt.tight_layout() 
    fig.subplots_adjust(top=0.85) 
    plt.show()
    


def plot_membrane_potential_path(model, data_path, target_neurons, title_addition=''):
    '''
        Plots the membrane potential dynamics for 'target_neurons'.

        Parameters: 
        @model: FC_SNN_torch instance. The model should have at least 3 hidden layers (including the output_layer)
        @data_path: .npz file containing the layer-wise activations / spike timestamps
        @target_neurons: can be a list or an integer representing the target neuron
            for which the membrane potential(s) should be plotted
        @title_addition: additional string title for plot
    '''
    
    # Convert integer into list if a list was not passed (simplifies later code)
    if not isinstance(target_neurons, list):
        target_neurons = [target_neurons]

    # read-in the data and convert into 
    with open(data_path, "rb") as f:
        activations = pickle.load(f)
        

    # Spike activations arriving from layer (N-1)
    input_activations = activations['layer_0']
    sorted_activations = np.sort(input_activations)
    sorted_indices = np.argsort(input_activations)

    # New output spikes generated in layer (N)
    output_activations = activations['layer_1']
    sorted_output_activations = np.sort(output_activations)

    # Get the spike threshold for the i-th neuron. In particular, 'D_i' is the trainable parameter that is unique to each neuron
    thresholds = {}
    for neuron in target_neurons:
        thresholds[neuron] = model.hidden_layers[1].t_max - model.hidden_layers[1].t_min - model.hidden_layers[1].D_i[neuron].detach().numpy()

    # For all neurons (i) in 'target_neurons':
    # Loop over a selection of neurons (j) from layer (N-1) which have sent a spike to neuron(i) in layer(N)
    # through the synapse (W_ji); collect spike times and membrane potentials for these neurons

    spike_times = {}
    potentials = {}
    for target_neuron_index in target_neurons:
        
        spike_times_i = []        # input spiking timestamps (tj) from layer (N-1) to neuron(i)
        potentials_i = []         # change in membrane potential neuron (i) in layer (N)
        for j in range(1,340,10):
            V_i = compute_membrane_potential(
                    sorted_activations[j],      # get next input spike time tj at which V should be evaluated
                    target_neuron_index,        
                    input_activations,         # all input spikes tj from layer (N-1)  
                    model.hidden_layers[1].kernel.data.detach().numpy()     # kernel W connecting layer (N-1) to layer (N)
                )

            spike_times_i.append(sorted_activations[j])
            potentials_i.append(V_i)

        spike_times[target_neuron_index] = spike_times_i 
        potentials[target_neuron_index] = potentials_i
    
    # print("Spike Times: ", spike_times)
    # print("Potentials: ", potentials)
    # breakpoint()

    # Start the plot 
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # --- Interval boundaries (t_min, t_max) --- #
    ax.axvline(x=model.hidden_layers[0].t_min, linestyle="--", color="black")
    ax.axvline(x=model.hidden_layers[0].t_max, linestyle="--", color="black")
    ax.axvline(x=model.hidden_layers[1].t_max, linestyle="--", color="black")

    
    
    # --- Threshold of layer (N) --- #
    ax.axhline(y=thresholds[target_neurons[0]], label=f"threshold", color='green', lw=4, alpha=0.7)

    # --- Membrane potential changes --- #
    colors = cm.tab10(np.linspace(0, 1, len(target_neurons))) 
    colors = dict(zip(target_neurons, colors))
    for i in target_neurons:
        ax.scatter(spike_times[i], potentials[i], alpha=0.5, color=colors[i])
    
    # --- Constant slope - fire phase --- #
    for i in target_neurons:
        x_spike = thresholds[i] + (model.hidden_layers[0].t_max - potentials[i][-1])
        y_spike = x_spike - model.hidden_layers[0].t_max + potentials[i][-1]
        ax.plot([model.hidden_layers[0].t_max, x_spike], 
                [potentials[i][-1], y_spike], 
                color=colors[i], alpha=0.5, lw=4,
                label=f"V_{i}")
    
    y_min, y_max = ax.get_ylim()
    output_spike_times = []
    # --- Generated Spike Output Times in layer(N) --- #
    for i in target_neurons:
        # get the spike times that were produced by each neuron after integration and threshold crossing
        output_spike_from_neuron = output_activations[i]
        output_spike_times.append(output_spike_from_neuron)
        ax.plot(output_spike_from_neuron, 0, color=colors[i], marker='^', markersize=15, label=f"Spike_{i}")
        ax.plot([output_spike_from_neuron, output_spike_from_neuron],
                [y_min, 0], linestyle="--", color=colors[i], alpha=0.5)


    # --- Axis labels --- #
    ax.set_xlabel('Spike Activation Time (t)', labelpad=20)
    ax.set_ylabel('Membrane Potential V(t)', labelpad=20)

    # Label the vertical lines for interval boundaries
    y_pos = (y_min) + (y_max - y_min) / 1.1     # position of (t_min) or (t_max) label
    ax.text(model.hidden_layers[0].t_min + 0.3, y_pos, 't_min_0', va='center', ha='left')
    ax.text(model.hidden_layers[0].t_max + 0.3, y_pos, 't_max_0\nt_min_1', va='center', ha='left')
    ax.text(model.hidden_layers[1].t_max + 0.3, y_pos, 't_max_1', va='center', ha='left')

    # xticks for generated spike times
    current_xticks = ax.get_xticks()
    new_xticks = sorted(set(current_xticks.tolist() + output_spike_times))
    ax.set_xticks(new_xticks)
    ax.set_xlim(0.0)

    
    title = 'Membrane Potential vs Spike Activations\nfor selected Neurons during Inference on a single MNIST image'
    if title_addition: title = title + "\n" + title_addition
    fig.suptitle(title)

    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))

    plt.tight_layout() 
    fig.subplots_adjust(top=0.85) 
    plt.show()

    

def plot_output_spikes(data_path, model=None, log_scale=False, relative_scale=False, additional_title=''):
    ''' Make a histogram plot to visualize the distribution of the input spike times layer-wise
        Requires a .npz file at 'data_path' with the logged spike times in a dict format, where 
        the keys correspond to a layer number such as 'layer_i' and the values in list format.
        #TODO: put model as 1st parameter and change function calls in main

        In case you see no spikes at all or spikes at unexpected times, make sure to 
        check that 'model.collect_activations' is being correctly set to True/False at any point
        where a forward pass might happen (training, evaluating, inference), and that after the 
        activations have been collected, they are stored in a .npz file e.g. using the function 
        'dump_activations' from the SNN model class
    '''
    if not os.path.isfile(data_path):
        config_utils.logging.info("### Could not find any logged files for plotting spike times - generate one first ###")
        return -1

    with open(data_path, "rb") as f:
        activations = pickle.load(f)
    
    # --- Layer-wise distributions
    total = 0
    plt.figure(figsize=(10, 5))
    print("\n\n PLOTTING ")
    layer_num = 0
    for k,v in activations.items():
        activations_np = np.round(np.array(v),decimals=2)
        layer_t_max = model.hidden_layers[layer_num].t_max
        print(f"layer_{k}.t_min={np.min(activations_np)}    max activations N={np.sum(activations_np == np.round(layer_t_max,2))}")

        # If the 'relative_scale' flag is enabled, the activations are plotted on an equivalent scale 
        # relative to the respective layer's t_max, rather than being plotted on an absolute time scale
        if relative_scale: activations_np = layer_t_max - activations_np

        # activations_np = activations_np[activations_np < layer_t_max]

        plt.hist(activations_np, bins=20, label=k, log=log_scale, alpha=0.6)
        total += len(activations_np)
        layer_num += 1

    # --- interval boundaries
    if model and not relative_scale:
        for i, layer in enumerate(model.hidden_layers):
            t_min=layer.t_min
            t_max=layer.t_max
            plt.axvline(x=t_min, color='r', linestyle='--', linewidth=1, alpha=0.7)
            plt.axvline(x=t_max, color='r', linestyle='--', linewidth=1, alpha=0.7)

            if i == 0: 
                plt.text(t_min + 0.2, plt.ylim()[1]*0.9, 't_min_0', va='top')
            elif i == len(model.hidden_layers)-1: 
                plt.text(t_min + 0.2, plt.ylim()[1]*0.9, f't_max_{i-1}\nt_min_{i}', va='top')
                plt.text(t_max + 0.2, plt.ylim()[1]*0.9, f't_max_{i+1}', va='top')
            else: 
                plt.text(t_min + 0.2, plt.ylim()[1]*0.9, f't_max_{i-1}\nt_min_{i}', va='top')

    
    plt.title(additional_title + f'\n\nDistribution of Spike Activations - N={total}')
    plt.xlabel('Spike Time Activation')
    plt.ylabel('Frequency')
    if relative_scale: plt.xlim(0, 8)
    else: plt.xlim(0, layer.t_max + 0.1*layer.t_max)
    plt.ylim(0, 340)
    plt.legend(loc='center left')
    plt.show()


def plot_input_tensor(t):
    plt.imshow(t, cmap='gray_r', interpolation='nearest')  # 'hot' = heatmap style
    plt.colorbar()
    plt.title("Heatmap of Encoded Tensor")
    plt.show()


def plot_input_spikes(input_tensor, additional_title=''):
    plt.figure(figsize=(10, 5))
    plt.hist(input_tensor, bins=20)
    plt.title(additional_title + '\n' + f'Distribution of Input Tensor Spikes')
    plt.xlabel('Spike Time')
    plt.ylabel('Frequency')
    plt.legend()
    plt.ylim(0,700)
    plt.show()

def plot_MNIST_tensor(original_image, tensor_input, label):
    if (original_image.shape[0]==1):
            original_image = original_image.squeeze()

    if (tensor_input.shape != (28,28)):
        if (tensor_input.shape[0]==1):
            tensor_input = tensor_input.squeeze()
        tensor_image = tensor_input.view(28,28)    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})

    # Plot original image
    im0 = ax[0].imshow(original_image, cmap='gray', aspect='equal')
    ax[0].set_title("Original MNIST image")
    ax[0].axis('off')

    # Plot processed image
    im1 = ax[1].imshow(tensor_image, cmap='gray_r', aspect='equal')  
    ax[1].set_title("TTFS Conversion- Heatmap")

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("Spike Time")

    # plt.tight_layout()
    fig.suptitle(f"Label: {label}", fontsize=14)
    plt.show()

def plot_discrete_tensor(path, delta_k, t_min, t_max):
    with open(path, 'r') as f:

        x = []
        y = []

        timestep_k = t_min
        timestep_count = 0
        total_spikes = 0
        
        for line in f:
            spikes_list = [int(x) for x in line.split(" ")] 
            active_indices = [i for i in range(len(spikes_list)) if spikes_list[i] == 1]
            total_spikes += len(active_indices)
            for active_index in active_indices: 
                x.append(timestep_k)
                y.append(active_index)
            timestep_k += delta_k
            timestep_count += 1

        neurons_number = len(spikes_list)

        plt.scatter(x,y,s=25, alpha=0.5)
        plt.xlim(t_min-0.05,t_max)
        plt.ylim(0, neurons_number)
        plt.title(f'Indices of Spiking Neurons\ndelta_k={delta_k} - discrete timesteps: K={timestep_count} - spike count: N={total_spikes}')
        plt.xlabel("Timestep K")
        plt.ylabel("Neuron Index")
        plt.show()

    return 0


def main():

    # Setup all network parameters by default
    robustness_params={
        'noise': 0.0,
        'time_bits': 0,
        'weight_bits': 0,
        'w_min': -1.0,
        'w_max': 1.0,
        'latency_quantiles': 1.0
    }

    # Argument parameters
    N_layers = 4
    data_name = 'MNIST'
    batch_size = 8
    model_name = 'FC4_noise'
    model_name = data_name + '-' + model_name
    model_type = 'SNN'
    add_image_noise = True 

    # Get Data Loaders
    dataset = dataset_torch.Dataset_Torch(
        data_name,
        batch_size,
        flatten= ('FC' in model_name),
        image_noise=add_image_noise,
        convert_ttfs = ('SNN' in model_type),   
    )

    # Also import the original dataset to get the input images without TTFS conversion - optionally, apply noise to the original images
    noise_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: dataset.add_strong_noise(x, enable=add_image_noise))
    ])
    image_dataset = datasets.MNIST(root='./datasets/MNIST', train=True, download=True, transform=noise_transforms)

    # Setup logging
    logging_dir = './logs/'
    config_utils.set_up_logging(logging_dir, model_name)

    # Get X_n ranges if available
    print(f"### X_n path={logging_dir + model_name + '_X_n.pkl'}")
    if os.path.exists(logging_dir + model_name + '_SNN_X_n.pkl'):
        X_n = pkl.load(open(logging_dir + model_name + '_SNN_X_n.pkl', 'rb'))
        config_utils.logging.info(f"### Loading X_n ranges from SNN - X_n={X_n}")

    elif os.path.exists(logging_dir + model_name + '_X_n.pkl'):
        X_n = pkl.load(open(logging_dir + model_name + '_X_n.pkl', 'rb'))
        config_utils.logging.info(f"### Loading X_n ranges from ANN - X_n={X_n}")
    
    else:
        X_n = [10,50] 
        config_utils.logging.info(f"### Loading default X_n - X_n={X_n}")

    # Get model instance
    model = create_torch_fc_model_SNN(X_n=X_n, layers=N_layers, robustness_params=robustness_params)
    print(model)

    # Load SNN weights directly if available
    if os.path.exists(logging_dir + model_name + '_full_snn_weights.pth'):
        config_utils.logging.info(f"### Loading full from-scratch SNN weights from {logging_dir + model_name + '_full_snn_weights.pth'}")
        load_path = logging_dir + model_name + '_full_snn_weights.pth'
        model.load_state_dict(torch.load(load_path, weights_only=True))
        model.eval()
    else: 
        config_utils.logging.info("### Cannot find pre-trained SNN model. Train SNN first and save. Try:\n " \
        "python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2_10  --testing=True --epochs=3 --layers=3 --save=True") # TODO 
        exit(1)

    # Setup model parameters
    model.set_snn_intervals(0,1)
    interval_bounds_list = []
    for i,layer in enumerate(model.hidden_layers):
        print(f"layer_{i}: t_min={layer.t_min} -- t_max={layer.t_max}")
        interval_bounds_list.append(np.round(layer.t_max,2))

    # Validate on testset
    config_utils.logging.info("Evaluation on testset")
    acc, _ = evaluate_FC_SNN(model, dataset.test_load)

    # Try forward pass
    model.collect_activations = True
    original_image, label = image_dataset[0]
    x, label = dataset.train_set[0]
    config_utils.logging.info(f"Shape of input x: {(x.shape)}")
    y = model(x)
    config_utils.logging.info(f"Model output: {y}")

    print("--- Frequency counts in raw image ---")
    orig_img_np = np.round(original_image.numpy(),2)
    values, counts = np.unique(orig_img_np, return_counts=True)
    frequency_dict = dict(zip(values, counts))
    print(frequency_dict)

    print("\n\n --- Frequency counts in TTFS input --- ")
    x_np = np.round(x.numpy(),2)
    print(np.unique(x_np, return_counts=True))

    plot_MNIST_tensor(original_image, x, label)
    plot_input_spikes(x, additional_title='Gray-Noise MNIST Five Image')

    act_path = logging_dir + 'outputs/' + model_name + '_plotting.npz'
    model.dump_activations(act_path)
    add_title = f'Forward Pass Noisy Input + re-trained model\nHidden Layers={len(model.hidden_layers)} | Test Acc.={acc}% | Intervals={interval_bounds_list}'
    plot_output_spikes(act_path, model=model, additional_title=add_title)



    return 0

if __name__ == "__main__":
    main()