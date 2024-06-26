import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from mnist import MNIST

# Load MNIST data

mndata = MNIST('.')
images, labels = mndata.load_training()
images_test, labels_test = mndata.load_testing()
images = np.array(images)  # Images are 28x28, already flattened to 784
images_test = np.array(images_test)
labels = np.array(labels)  # Convert to numpy array
labels_test = np.array(labels_test)
print(f"images.shape: {images.shape}")
print(f"labels.shape: {labels.shape}")
print(f"images_test.shape: {images_test.shape}")
print(f"labels_test.shape: {labels_test.shape}")

# Select random subset of data
n_samples = 100
np.random.seed(4)  # Set random seed for reproducibility
random_indices = np.random.choice(images.shape[0], n_samples, replace=False)
images = images[random_indices]
labels = labels[random_indices]
print(f"Subset images.shape: {images.shape}")
print(f"Subset labels.shape: {labels.shape}")

# One-hot encode the labels
labels = np.eye(10)[labels]
labels_test = np.eye(10)[labels_test]
print(f"labels.shape: {labels.shape}")
print(f"Head of labels (check 1-hot encoding): {labels[:5]}")

# Define Lasso loss and objective functions

def loss(y, x, beta):
    """Loss function
    
    Parameters
    ----------
    y : numpy.ndarray
        Dependent variable
    x : numpy.ndarray
        Independent variable
    beta : numpy.ndarray
        Coefficients

    Returns
    -------
    float
        Loss
    """
    return cp.sum_squares(y - x @ beta)

def l1_regularization(beta, lambda1):
    """L1 regularization function
    
    Parameters
    ----------
    beta : numpy.ndarray
        Coefficients
    lambda1 : float
        L1 regularization parameter

    Returns
    -------
    float
        L1 regularization
    """
    return lambda1 * cp.norm1(beta)

def lasso_objective(y, x, beta, lambda1):
    """Lasso loss function

    Equivalent to elastic net loss with lambda2 = 0

    Parameters
    ----------
    y : numpy.ndarray
        Dependent variable
    x : numpy.ndarray
        Independent variable
    beta : numpy.ndarray
        Coefficients
    lambda1 : float 
        L1 regularization parameter

    Returns
    -------
    float
        Lasso loss
    """
    return loss(y, x, beta) + l1_regularization(beta, lambda1)

# Create and solve the optimization problem

n = images.shape[0]
d = images.shape[1]
beta = cp.Variable((d, 10))
lambda1 = 1.0
y = labels
x = images
objective = lasso_objective(y, x, beta, lambda1)
problem = cp.Problem(cp.Minimize(objective))
problem.solve(solver=cp.CLARABEL)
print(f"Optimal beta: {beta.value}")

# Calculate accuracy on test set

def decode_one_hot(one_hot):
    return np.argmax(one_hot, axis=-1)

y_test = labels_test
x_test = images_test
predictions = x_test @ beta.value
print(f"predictions.shape: {predictions.shape}")
accuracy = np.mean(decode_one_hot(predictions) == decode_one_hot(y_test))
print(f"Test accuracy: {100*accuracy:.3g} percent")

# Print the first 10 predictions and true labels
print("First 10 predictions:")
print(np.argmax(predictions[:10], axis=1))
print("True labels:")
print(np.argmax(y_test[:10], axis=1))

# Decode beta values to images and plot the matrices

def decode_beta(beta):
    return beta.reshape(28, 28, 10)

beta_images = decode_beta(beta.value)
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
max_val = np.max(np.abs(beta_images))
min_val = -max_val
for i in range(10):
    if i != 9:
        ax[i//5, i%5].imshow(beta_images[:, :, i], vmin=min_val, vmax=max_val, cmap='bwr')
    else:
        plt.colorbar(ax[i//5, i%5].imshow(beta_images[:, :, i], vmin=min_val, vmax=max_val, cmap='bwr'), ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title(f"Digit {i}")
plt.tight_layout()
plt.show()