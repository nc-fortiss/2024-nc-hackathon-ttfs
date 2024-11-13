import numpy as np
import tensorflow as tf
import cv2  # For image processing (install with `pip install opencv-python` if needed)
import os
import pandas as pd


def load_gtsrb_data(data_dir='./data', img_size=(32, 32)):
    
    def load_images_and_labels(directory, img_size=(32, 32), is_train=True):
        images, labels = [], []
        
        if is_train:
            # Loop through each subdirectory in the training directory
            for label_dir in os.listdir(directory):
                label_path = os.path.join(directory, label_dir)
                if os.path.isdir(label_path):
                    # Define the path to the class annotation CSV file for the folder
                    csv_file = os.path.join(label_path, f'GT-{label_dir}.csv')
                    
                    # Ensure the CSV file exists before attempting to read it
                    if not os.path.isfile(csv_file):
                        print(f"Warning: CSV file {csv_file} not found, skipping directory {label_dir}")
                        continue
                    
                    # Read annotations from the CSV file
                    annotations = pd.read_csv(csv_file, sep=';')
                    
                    # Load each image using its filename from the CSV and its class label
                    for _, row in annotations.iterrows():
                        img_file = row['Filename']
                        class_id = row['ClassId']
                        img_path = os.path.join(label_path, img_file)
                        
                        # Attempt to read the image file
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, img_size)  # Resize to target size
                            images.append(img)
                            labels.append(class_id)
                        else:
                            print(f"Warning: Unable to load image at {img_path}")
        else:
            # For test data, use a single CSV file named "GT-final_test.csv"
            csv_file = os.path.join(directory, 'GT-final_test.csv')
            
            # Ensure the CSV file exists before attempting to read it
            if not os.path.isfile(csv_file):
                print(f"Error: Test CSV file {csv_file} not found.")
                return np.array(images), np.array(labels)
            
            # Read annotations from the CSV file
            annotations = pd.read_csv(csv_file, sep=';')
            
            # Load each image using its filename from the CSV and its class label
            for _, row in annotations.iterrows():
                img_file = row['Filename']
                class_id = row['ClassId']
                img_path = os.path.join(directory, img_file)
                
                # Attempt to read the image file
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)  # Resize to target size
                    images.append(img)
                    labels.append(class_id)
                else:
                    print(f"Warning: Unable to load image at {img_path}")

        # Convert lists to numpy arrays after loading
        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int')
        return images, labels

    # Load and preprocess training and testing data
    x_train, y_train = load_images_and_labels(os.path.join(data_dir, 'training'), is_train = True)
    x_test, y_test = load_images_and_labels(os.path.join(data_dir, 'test'), is_train = False)

    # Normalize images to the [0, 1] range
    x_train /= 255.0
    x_test /= 255.0 # in standard 8-bit color images, pixel values range from 0 to 255 for each color channel (Red, Green, and Blue)

    return x_train, y_train, x_test, y_test


class Dataset:
    def __init__(
        self,
        data_name,
        logging_dir,
        flatten,
        ttfs_convert,
        ttfs_noise=0,
    ):
        self.name = data_name
        self.flatten=flatten
        self.noise=ttfs_noise
        self.logging_dir=logging_dir
        # Load original data.
        self.get_features_vectors()
        # In case of SNN, convert input data with TTFS coding.
        self.ttfss_convert=ttfs_convert
        if ttfs_convert: self.convert_ttfs()
        
    def get_features_vectors(self):
        """
        Load image datasets and transform into features. 
        """
        if 'MNIST' in self.name:
            self.input_shape, self.train_sample=(28, 28, 1), 1/64
            self.q, self.p = 1.0, 0.0
            self.num_of_classes = 10
            if self.name=='MNIST':
                (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            else:
                (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
            self.x_train, self.x_test = self.x_train/255.0, self.x_test/255.0
            if self.flatten:
                self.x_train, self.x_test = self.x_train.reshape((len(self.x_train), -1)), self.x_test.reshape((len(self.x_test), -1))
            else:
                self.x_train, self.x_test = self.x_train.reshape(-1, 28, 28, 1), self.x_test.reshape(-1, 28, 28, 1)
        elif 'CIFAR' in self.name:
            # CIFAR10 or CIFAR100 dataset.
            self.input_shape=(32, 32, 3)
            self.q, self.p = 3.0, -3.0
            if self.name=='CIFAR10':
                # CIFAR10 dataset.
                self.num_of_classes = 10
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar10.load_data()
                # Mean and std to scale input.
                self.mean_test, self.std_test=120.707, 64.15
            else:
                # CIFAR100
                self.num_of_classes = 100
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar100.load_data()
                # Mean and std to scale input.
                self.mean_test, self.std_test=121.936, 68.389
            # Scale to [-3, 3] range.
            self.x_test, self.x_train=(self.x_test-self.mean_test)/(self.std_test+1e-7), (self.x_train-self.mean_test)/(self.std_test+1e-7)
        elif 'GTSRB' in self.name:
            self.input_shape = (32, 32, 3)
            self.q, self.p = 3.0, -3.0
            self.num_of_classes = 43
            self.x_train, self.y_train, self.x_test, self.y_test = load_gtsrb_data()
            
            if self.x_train.size == 0 or self.x_test.size == 0:
                raise ValueError("Training or test data is empty. Please check the data loading function and paths.")

            self.mean_test = np.mean(self.x_train, axis=(0, 1, 2))
            self.std_test = np.std(self.x_train, axis=(0, 1, 2))
            
            # Normalize to [-3, 3] range ??
            self.x_train = (self.x_train - self.mean_test) / (self.std_test + 1e-7)
            self.x_test = (self.x_test - self.mean_test) / (self.std_test + 1e-7)
    
        self.x_train, self.x_test = self.x_train.astype('float64'), self.x_test.astype('float64')
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_of_classes) 
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_of_classes)
        print ('Train data:', np.shape(self.x_train), np.shape(self.y_train))
        print ('Test data:', np.shape(self.x_test), np.shape(self.y_test))

    def convert_ttfs(self):
        """
        Convert input values into time-to-first-spike spiking times.
        """
        self.x_test, self.x_train = (self.x_test - self.p)/(self.q-self.p), (self.x_train - self.p)/(self.q-self.p)
        self.x_train, self.x_test=1 - np.array(self.x_train), 1 - np.array(self.x_test)
        self.x_test=np.maximum(0, self.x_test + tf.random.normal((self.x_test).shape, stddev=self.noise, dtype=tf.dtypes.float64))
