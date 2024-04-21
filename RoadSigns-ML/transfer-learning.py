# TRANSFER LEARNING

import numpy as np
import pvml
import matplotlib.pyplot as plt
import os
from skimage import exposure

# Load the pre-trained CNN and MLP models
cnn = pvml.CNN.load("pvmlnet.npz")
mlp = pvml.MLP.load("road-signs-mlp.npz")

# Combine the weights and biases of the two models
cnn.weights[-1] = mlp.weights[0][None, None, :, :]
cnn.biases[-1] = mlp.biases[0]

# Save the combined model
cnn.save("road-signs-cnn.npz")

# Path and image file to test
imagepath = "road-signs/test/misc-parking/03.jpg"

# Load and preprocess the image
image = plt.imread(imagepath) / 255
#image = image.sum(2)
#image = np.stack([image, image, image], -1)  # Convert grayscale to RGB
image = exposure.equalize_hist(image)

# Run inference on the image using the combined model
labels, probs = cnn.inference(image[None, :, :, :])

# Get the class labels
classes = os.listdir("road-signs/test")
classes = [c for c in classes if not c.startswith(".")]
classes.sort()

# Display the top 5 predicted classes with their probabilities
indices = (-probs[0]).argsort()
for k in range(5):
     index = indices[k]
     print(f"{k+1} {classes[index]:10} {probs[0][index] * 100: .1f}")

# Display the image
plt.imshow(image)
plt.show()


# Function to create a confusion matrix
def make_confusion_matrix_calc_display(path, cnn, name):
    confusionMatrix = np.zeros((len(classes), len(classes)), dtype=float)
    incorrect = 0
    total = 0
    
    # Iterate through the classes and images in the test path
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
                        
            #image = image.sum(2)
            #image = np.stack([image, image, image], -1)  # Convert grayscale to RGB
            image = exposure.equalize_hist(image)
            
            # Run inference on the image
            labels, probs = cnn.inference(image[None, :, :, :])
            
            # Update the confusion matrix
            confusionMatrix[classes.index(klass)][labels[0]] += 1
            
            # Check if the prediction is incorrect
            if classes.index(klass) != labels[0]:
                incorrect += 1
            
            total += 1
    
    # Calculate accuracy
    accuracy = ((total - incorrect) / total) * 100
    print(name + " accuracy is: ", accuracy)
    
    # Normalize the confusion matrix
    confusionMatrix = confusionMatrix / confusionMatrix.sum(1, keepdims=True)
    
    # Display the confusion matrix
    plt.figure(figsize=(30, 30))
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(name + " Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = confusionMatrix.max() / 2.0
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            if confusionMatrix[i, j] == 0:
                text = '0'
            else:
                text = format(confusionMatrix[i, j], '.1')
                print(text)
            plt.text(j, i, text,
                     ha="center", va="center",
                     color="white" if confusionMatrix[i, j] > thresh else "black")
    
    # Save and show the confusion matrix
    plt.savefig(name + "confusion")
    plt.show()

# Create a confusion matrix and calculate accuracy for the test dataset
make_confusion_matrix_calc_display("road-signs/test", cnn, "first-try")