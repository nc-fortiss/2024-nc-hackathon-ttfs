
import numpy as np
from model_torch import compute_membrane_potential
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

def plot_membrane_potential(model, target_neurons):
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
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))

    # --- Interval boundaries (t_min, t_max) --- #
    ax.axvline(x=model.hidden_layers[0].t_min, linestyle="--", color="gray")
    ax.axvline(x=model.hidden_layers[0].t_max, linestyle="--", color="gray")
    ax.axvline(x=model.hidden_layers[1].t_max, linestyle="--", color="gray")

    
    
    # --- Threshold of layer (N) --- #
    print(thresholds)
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

    # --- Generated Spike Output Times in layer(N) --- #
    for i in target_neurons:
        # get the spike times that were produced by each neuron after integration and threshold crossing
        output_spike_from_neuron = model.layer_activations[1][i]
        ax.plot(output_spike_from_neuron, 0, color=colors[i], marker='^', markersize=15, label=f"Spike_{i}")


    # --- Axis labels --- #
    ax.set_xlabel('Spike Activation Time (t)', labelpad=20)
    ax.set_ylabel('Membrane Potential V(t)', labelpad=20)

    y_min, y_max = ax.get_ylim()
    y_pos = (y_min) + (y_max - y_min) / 1.1     # position of (t_min) or (t_max) label
    ax.text(model.hidden_layers[0].t_min + 1, y_pos, 't_min_0', va='center', ha='left')
    ax.text(model.hidden_layers[0].t_max + 1, y_pos, 't_max_0\nt_min_1', va='center', ha='left')
    ax.text(model.hidden_layers[1].t_max + 1, y_pos, 't_max_1', va='center', ha='left')
    ax.set_title('Membrane Potential vs Spike Activations\nSelected Neurons during Inference on a single MNIST image', pad=30)

    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))

    plt.tight_layout() 
    plt.show()