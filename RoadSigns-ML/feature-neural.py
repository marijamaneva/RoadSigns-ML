# FEATURE EXTRACTION using neural network - Neural features 

import numpy as np
import matplotlib.pyplot as plt
import os 
import pvml
from skimage import exposure

# listing the content of the directories; these will represent the classes
classes = os.listdir("road-signs/train")
classes = [c for c in classes if not c.startswith(".")]
classes.sort()

# Function to extract neural features from an image using a CNN
def extract_neural_features(im, cnn):
    activations = cnn.forward(im[None,:,:,:])
    features = activations[-3]                     # Extracting activations from the third last layer
    if len(features.shape) > 2:
        features = np.mean(features, axis=(1, 2))  # Average over the spatial dimensions
    features = features.reshape(-1)
    return features

# Function that will extract and read the images from the specified folder
def process_directory(path, cnn):
    all_features = []       # list to store extracted features
    all_labels = []         # list to store corresponding labels
    klass_label = 0         # label for the current class
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        image_files = [c for c in image_files if not c.startswith(".")]
        # read each image in the class
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename 
            image = plt.imread(image_path) / 255.0            # read and normalize the image
            #image = image.sum(2)
            #image = np.stack([image, image, image], -1)  # Gray to RGB
            
            # Apply histogram equalization
            image = exposure.equalize_hist(image)
            
            print(image_path)
            
            features = extract_neural_features(image, cnn)    # extract neural features using the provided CNN
            #features = features.reshape(-1)                  # flatten the features
            all_features.append(features)                     # add features to the list
            all_labels.append(klass_label)                    # add label to the list
        klass_label += 1 
    X = np.stack(all_features, 0)                             # stack the features into an array
    Y = np.array(all_labels)                                  # convert labels to numpy array
    return X, Y

# Load the pre-trained CNN model
cnn = pvml.CNN.load("pvmlnet.npz")

# Process the images in the "test" directory
X, Y = process_directory("road-signs/test", cnn)
print("test", X.shape, Y.shape)
data = np.concatenate([X,Y[:,None]], 1)      # concatenate features and labels
np.savetxt("test.txt.gz", data)              # save the data to a text file

# Process the images in the "train" directory
X, Y = process_directory("road-signs/train", cnn) 
print("train", X.shape, Y.shape) 
data = np.concatenate([X,Y[:,None]], 1)     # concatenate features and labels
np.savetxt("train.txt.gz", data)        

