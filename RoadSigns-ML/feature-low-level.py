# FEATURE EXTRACTION with low level features

import numpy as np
import matplotlib.pyplot as plt
import os 
import image_features
from skimage import exposure


# listing the content of the directories; these will represent the classes
classes = os.listdir("road-signs/train")
classes = [c for c in classes if not c.startswith(".")]


# function that will extract and read the images from the specified folder
def process_directory(path):
    all_features = []       # list to store extracted features
    all_labels = []         # list to store corresponding labels
    klass_label = 0         # label for the current class
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        image_files = [c for c in image_files if not c.startswith(".")]
        # read each image in the class
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename 
            image = plt.imread(image_path) / 255.0                      # read and normalize the image
            #image = image.sum(2)
            #image = np.stack([image, image, image], -1)  # Gray to RGB
            print(image_path)
            
            # Extract the first set of features from the image
            features1 = image_features.color_histogram(image)
            features1 = features1.reshape(-1)                              # flatten the features
            # Apply histogram equalization to the first set of features
            features1 = exposure.equalize_hist(features1)
 
            # Extract the second set of features from the image
            features2 = image_features.edge_direction_histogram(image)
            features2 = features2.reshape(-1)                               # flatten the features
            # Apply histogram equalization to the second set of features
            features2 = exposure.equalize_hist(features2)
            
            # Concatenate the feature vectors
            combined_features = np.concatenate([features1, features2])
                         
            all_features.append(combined_features)                         # add features to the list
            all_labels.append(klass_label)                                 # add label to the list
        klass_label += 1 
    X = np.stack(all_features, 0)                                          # stack the features into an array
    Y = np.array(all_labels)                                               # convert labels to numpy array
    return X, Y


# process the images in the "test" directory
X, Y = process_directory("road-signs/test")
print("test", X.shape, Y.shape)
data = np.concatenate([X,Y[:,None]], 1)     # concatenate features and labels
np.savetxt("test.txt.gz",data)              # save the data to a text file

# process the images in the "train" directory
X, Y = process_directory("road-signs/train")
print("train", X.shape, Y.shape)
data = np.concatenate([X,Y[:,None]], 1)     # concatenate features and labels
np.savetxt("train.txt.gz",data)             # save the data to a text file



