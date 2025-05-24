import h5py
import re

file_path = "./logs/CIFAR10-VGG_nobatch_example_preprocessed.h5"

def extract_index(key):
    """Extracts the numeric part from layer names like 'conv2d_3/kernel:0' → 3"""
    match = re.search(r'(\d+)', key)
    return int(match.group()) if match else float('inf')

with h5py.File(file_path, "r") as f:
    keys = list(f.keys())
    
    # Sort by numeric index (e.g. conv2d_1, conv2d_2, ...)
    sorted_keys = sorted(keys, key=extract_index)
    
    for k in sorted_keys:
        print(k)

    
def extract_index(key):
    """Extracts the numeric part from layer names like 'conv2d_3/kernel:0' → 3"""
    match = re.search(r'(\d+)', key)
    return int(match.group()) if match else float('inf')


print("\n\n\n")
with h5py.File(file_path, "r") as f:
    keys = list(f.keys())
    
    # Sort by numeric index (e.g. conv2d_1, conv2d_2, ...)
    sorted_keys = sorted(keys, key=extract_index)
    
    for k in sorted_keys:
        print(k)

print("\n\n\n")
with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
            # Optional: print actual values (comment out if large)
            # print(obj[:])
            if 'pool' in name: 
                print(obj[:])
            print("\n")
    f.visititems(print_structure)
