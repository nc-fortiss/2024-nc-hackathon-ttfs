import Dataset 
from pytorch_src import dataset_torch
import unittest
import numpy as np 
import os

class TestDataset(unittest.TestCase):


    def test_FC_ReLU_input(self):
        ''' Instantiates the dataset objects for tensorflow and pytorch; 
            picks a (transformed and converted) input tensor from a random index training sample
            and checks if the tensors are similar/close/equal for both tf and torch.
        '''
        # flatten=True because of 'FC' model; ttfs_convert=False since normal ANN
        tf_dataset = Dataset.Dataset("MNIST", "./logs", flatten=True,ttfs_convert=False,ttfs_noise=0.0)
        torch_dataset = dataset_torch.Dataset_Torch("MNIST",flatten=True,convert_ttfs=False,ttfs_noise=0.0)

        rand_index = np.random.randint(0, tf_dataset.x_train.shape[0])

        # Tensorflow dataset distinguishes between input X and target y variables
        tf_tensor = tf_dataset.x_train[rand_index]

        # torch dataset can only access one training tuple at a time - index 0 is the input X tensor
        torch_tensor = torch_dataset.train_set[rand_index][0].numpy()

        similarity = np.allclose(torch_tensor, tf_tensor, atol=1e-8, rtol=1e-7)
        self.assertEqual(similarity, True)

    def test_FC_SNN_input(self):
        ''' Instantiates the tf and torch dataset objects, with all the necessary transforms 
            for the data to be input into the FC SNN and compares a training sample tensor at a random index
        '''

        tf_dataset = Dataset.Dataset("MNIST", "./logs", flatten=True,ttfs_convert=True,ttfs_noise=0.0)
        torch_dataset = dataset_torch.Dataset_Torch("MNIST",flatten=True,convert_ttfs=True,ttfs_noise=0.0)

        rand_index = np.random.randint(0, tf_dataset.x_train.shape[0])

        # Tensorflow dataset distinguishes between input X and target y variables
        tf_tensor = tf_dataset.x_train[rand_index]

        # torch dataset can only access one training tuple at a time - index 0 is the input X tensor
        torch_tensor = torch_dataset.train_set[rand_index][0]

        similarity = np.allclose(torch_tensor, tf_tensor, atol=1e-8, rtol=1e-7)
        self.assertEqual(similarity, True)

        np.savetxt('./tf', tf_tensor, delimiter=',')
        np.savetxt('./torch',torch_tensor, delimiter=',')


if __name__ == "__main__":
    unittest.main()