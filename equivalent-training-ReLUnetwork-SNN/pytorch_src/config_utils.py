import os 
import logging 
import sys
import matplotlib.pyplot as plt


'''
    Module containing global configuration settings and logging / utility functions
'''


DEBUG_MODE = False
LOGGING_DIR = ''

def set_up_logging(logging_dir, model_name):
    """
    Set up logging for the simulation. Taken from the tensorflow implementation
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
           logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
           logging.StreamHandler(sys.stdout),
        ],
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    
    # Set the logging_dir variable
    global LOGGING_DIR
    LOGGING_DIR = logging_dir

    
def clean_spike_logs():
    ''' Cleans any existing spike-time files from the logging directory  '''
    for file in os.listdir(LOGGING_DIR):
        if file.startswith('spike_output'):
            file_path = os.path.join(LOGGING_DIR, file)
            os.remove(file_path)  
            logging.info(f"### Removed {file_path} ###")


def plot_input_spikes():
    ''' Make a histogram plot to visualize the distribution of the input spike times layer-wise
        Requires a 'spike_output.txt' logging file to have been generated (e.g. during a single forward pass)
    '''

    spikes_file_path = os.path.join(LOGGING_DIR, 'spike_output.txt')

    if not os.path.isfile(spikes_file_path):
        logging.info("### Could not find any logged files for plotting spike times - generate one first ###")
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

    # plt.xlim(left = 0)
    plt.show()


    return 0
