
import numpy as np
from model_torch import compute_membrane_potential
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import config_utils
import os 
from model_torch import *
from train_torch import * 
import dataset_torch
import pickle as pkl 

def plot_membrane_potential(model, target_neurons, title_addition=''):
    '''
        Plots the membrane potential dynamics for 'target_neurons'.

        Arguments:
        @model: FC_SNN_torch instance. The model should have at least 3 hidden layers (including the output_layer)
        @target_neurons: can be a list or an integer representing the target neuron
            for which the membrane potential(s) should be plotted
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



def plot_input_spikes():
    ''' Make a histogram plot to visualize the distribution of the input spike times layer-wise
        Requires a 'spike_output.txt' logging file to have been generated (e.g. during a single forward pass)
    '''

    spikes_file_path = os.path.join(config_utils.LOGGING_DIR, 'spike_output.txt')

    if not os.path.isfile(spikes_file_path):
        config_utils.logging.info("### Could not find any logged files for plotting spike times - generate one first ###")
        return -1

    spikes_per_layer = []

    # convert input strings into a list of lists (list of spike times per layer)
    # each line in the input corresponds to the spike values of a single layer
    with open(spikes_file_path, 'r') as f:
        for line in f:
            line = line.strip("\n")  
            spikes = [float(v) for v in line.split()]  
            spikes_per_layer.append(spikes)


    plt.figure(figsize=(10, 5))
    # plt.hist(spikes_per_layer, bins=30, density=True)

    plt.hist(spikes_per_layer[0], bins=80, density=True)
    # plt.hist(spikes_per_layer[1], range=(2998, 3003), bins=30, density=True)

    plt.title('Spike Time Activation Distribution for converted Input')
    plt.xlabel('Spike Time Activation')
    plt.ylabel('Frequency')

    # plt.xlim(left = 0)
    plt.show()


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
    N_layers = 3
    data_name = 'MNIST'
    batch_size = 8
    model_name = 'FC2'
    model_type = 'SNN'

    # Get Data Loaders
    dataset = dataset_torch.Dataset_Torch(
        data_name,
        batch_size,
        flatten= ('FC' in model_name),
        convert_ttfs = ('SNN' in model_type),   
    )

    # Setup logging
    logging_dir = './logs/'
    config_utils.set_up_logging(logging_dir, model_name)

    # Get X_n ranges if available
    if os.path.exists(logging_dir + model_name + '_X_n.pkl'):
        X_n = pkl.load(open(logging_dir + model_name + '_X_n.pkl', 'rb'))
    else:
        X_n = [10,50] 

    # Get model instance
    model = create_torch_fc_model_SNN(X_n=X_n, layers=N_layers, robustness_params=robustness_params)
    print(model)

    # Load SNN weights directly if available
    if os.path.exists(logging_dir + 'model/full_snn_weights.pth'):
        config_utils.logging.info("### Loading full from-scratch SNN weights")
        load_path = logging_dir + 'model/full_snn_weights.pth'
        model.load_state_dict(torch.load(load_path, weights_only=True))
        model.eval()
    else: 
        config_utils.loggin.info("### Cannot find pre-trained SNN model. Train SNN first and save. Try:\n " \
        "python3 main_torch.py --data_name=MNIST --model_type=SNN --model_name=FC2_10  --testing=True --epochs=3 --layers=3 --save=True") # TODO 
        exit(1)

    # Setup model parameters
    model.set_snn_intervals(0,1)

    # Try forward pass
    tuple = dataset.train_set.__getitem__(0)
    x = tuple[0]
    config_utils.logging.info(f"Shape of input x: {(x.shape)}")
    y = model(x)
    config_utils.logging.info(f"Model output: {y}")

    # Validate on testset
    config_utils.logging.info("Evaluation on testset")
    evaluate_FC_SNN(model, dataset.test_load)

    # 




    return 0

if __name__ == "__main__":
    main()