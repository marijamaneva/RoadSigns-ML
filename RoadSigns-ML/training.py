# Training data

import numpy as np
import matplotlib.pyplot as plt
import pvml

# Load the training data from "train.txt.gz" and separate features (X) from labels (Y)
data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1].astype(int)

# Load the test data from "test.txt.gz" and separate features (Xtest) from labels (Ytest)
data = np.loadtxt("test.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)

# Determine the number of classes in the data
nclasses = Y.max() + 1

# Create an MLP model with the number of input features and output classes
mlp = pvml.MLP([X.shape[1], nclasses])

# Initialize variables for storing accuracy and epoch information
plt.ion()  # Enable interactive mode for plotting
train_accs = []
test_accs = []
epochs = []

# Training loop
for epoch in range(20000):
    steps = X.shape[0] // 50  # Determine the number of steps for training batches
    
    # Train the MLP model on the training data
    mlp.train(X, Y, lr=0.001, batch=50, steps=steps)
    
    # Evaluate the model's accuracy on training and test data at regular intervals
    if epoch % 1000 == 0:
        # Calculate predictions and probabilities for the training data
        predictions, probs = mlp.inference(X)
        train_acc = (predictions == Y).mean()  # Compute training accuracy
        
        # Calculate predictions and probabilities for the test data
        predictions, probs = mlp.inference(Xtest)
        test_acc = (predictions == Ytest).mean()  # Compute test accuracy
        
        # Print and store the training and test accuracy
        print(f"{epoch} {train_acc * 100 : .1f} {test_acc * 100 : .1f}")
        train_accs.append(train_acc * 100)
        epochs.append(epoch)
        test_accs.append(test_acc * 100)
        
        # Plot the training and test accuracy over epochs
        plt.clf()
        plt.plot(epochs, train_accs)
        plt.plot(epochs, test_accs)
        plt.legend(["train", "test"])
        plt.xlabel("Number of epochs")
        plt.ylabel("Probability %")
        plt.pause(0.01)
        
# Save the trained MLP model
mlp.save("road-signs-mlp.npz")

# Turn off interactive mode and display the plot
plt.ioff()
plt.show()



