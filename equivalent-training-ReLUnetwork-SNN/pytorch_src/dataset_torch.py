import numpy as np
import torch
import torch.utils
from torchvision import datasets, transforms


class Dataset_Torch:
    ''' Creates and returns train and test data, in the proper format, shape and values by importing from torch datasets

    Attributes:
        name: the name of the dataset to use (MNIST, CIFAR, ...)
        flatten: boolean, if True then the input tensor is flattened to fit the 1st layer
        ttfs_noise: TODO
        convert_ttfs: boolean flag, if True then the input pixel values are converted into TTFS spikes
        input_shape: shape of the original input
        train_sample: TODO
        q,p: TODO
        num_of_classes: total number of distinct output labels for classification
        train_set, test_set: contain the training and testset datasets (including both features and labels)
        train_load, test_load: provides the respective DataLoader interface for accessing batches
    '''

    def __init__(self, dataset_name, flatten, convert_ttfs, ttfs_noise=0, ):
        self.name = dataset_name
        self.flatten = flatten  
        self.ttfs_noise = ttfs_noise 
        self.convert_ttfs = convert_ttfs

        self.get_features_vectors()         # TODO: pass 'flatten' as a variable instead of setting it as an attribute (?)
        self.convert_ttfs = convert_ttfs

        self.train_load = torch.utils.data.DataLoader(self.train_set, batch_size=64,shuffle=True)
        self.test_load = torch.utils.data.DataLoader(self.test_set, batch_size=64,shuffle=False)

      
    def get_features_vectors(self):
        """
        Load image datasets and turn pixels into features by applying the proper transforms. 
        """
        ''' A series of lambda functions to be used as transform compositions on the input data '''
        def conditional_flatten(x, flatten):
            ''' Re-shape and/or flatten the input if fully connected layer is first '''
            if flatten:
                return x.reshape(-1)  # Flatten to 1-D vector for fully connected input layer
            else: 
                return x.reshape(28, 28, 1)   # Add grayscale dimension
            
        def convert_ttfs_fun(x):
            ''' Convert input pixel values into time-to-first-spike spiking times. '''
            # TODO: apply noise to test data
            x = (x - self.p) / (self.q - self.p)
            x = 1 - np.array(x)
            x = torch.tensor(x,dtype=torch.float64)
            return x 
        
        def add_noise(x, noise):
            ''' Add random noise to the input of the test set '''

            x = x + torch.normal(mean=0.0, std=noise, size=x.shape)
            x = np.maximum(0, x)
            return x 


        def to_float_64(x):
            return x.to(dtype=torch.float64)
        

        if 'MNIST' in self.name:
            self.input_shape, self.train_sample=(28, 28, 1), 1/64
            self.q, self.p = 1.0, 0.0       # TODO: understand what p and q are
            self.num_of_classes = 10

            # Apply transforms and conversions directly in the data-loading step as opposed to the load, then convert approach as in tensorflow
            train_transform = transforms.Compose([
                transforms.ToTensor(),  # Converts (H, W) → (1, H, W) and normalizes to [0,1]
                transforms.Lambda(lambda x: to_float_64(x)), 
                transforms.Lambda(lambda x: conditional_flatten(x, self.flatten)),   # Re-shapes input tensors as needed
                transforms.Lambda(lambda x: convert_ttfs_fun(x) if self.convert_ttfs else  x)   # Converts pixels into spikes if needed
            ])

            test_transform = transforms.Compose(train_transform.transforms + [
                transforms.Lambda(lambda x: add_noise(x, self.ttfs_noise))  # Additional transform for test only
            ])

            if self.name=='MNIST':
                # 'download=True' downloads the data from internet, if not already done; 'train=True' specifies training set
                # 'root=PATH' specifies the directory where the dataset shall be saved 
                self.train_set = datasets.MNIST(root='./datasets/MNIST', train=True, download=True, transform=train_transform)
                self.test_set = datasets.MNIST(root='./datasets/MNIST', train=False, download=True, transform=test_transform)
            else:
                self.train_set = datasets.FashionMNIST(root='./datasets/FASHION_MNIST', train=True, download=True, transform=train_transform)
                self.test_set = datasets.FashionMNIST(root='./datasets/FASHION_MNIST', train=False, download=True, transform=test_transform)

                