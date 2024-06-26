import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Define the data

temperature = np.array([
    75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, \
    54, 56, 59, 63, 67, 72
])
n_data = len(temperature)
time = np.arange(1, n_data + 1)


# Visualize the data

fig, ax = plt.subplots()
ax.plot(time, temperature, '.')
# plt.show()

# Create n_samples random corrupted data samples from corrupting by 30 percent a single original data sample
# The first sample is the original temperature data
n_samples = 30
np.random.seed(4)  # Set random seed for reproducibility
y = np.zeros((n_samples, len(temperature)))
y[0] = temperature
for i in range(1, n_samples):
    y[i] = temperature
    jrand = np.random.randint(n_data)
    y[i, jrand] = y[i, jrand] * (1 + .3*np.array([-1,1])[np.random.randint(2)])

# Define elastic net loss and objective functions

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

def l2_regularization(beta, lambda2):
    """L2 regularization function
    
    Parameters
    ----------
    beta : numpy.ndarray
        Coefficients
    lambda2 : float
        L2 regularization parameter

    Returns
    -------
    float
        L2 regularization
    """
    return lambda2 * cp.norm2(beta)

def elastic_objective(y, x, beta, lambda1, lambda2):
    """Elastic net loss function

    Trades off between L1 and L2 regularization. The L1 regularization encourages
    sparsity in the coefficients, while the L2 regularization encourages small
    coefficients. The elastic net loss is the sum of the squared residuals and
    the L1 and L2 regularization terms.

    For a Lasso objective, set lambda2 = 0. For a Ridge penalty, set lambda1 = 0.
    For a standard least squares objective, set lambda1 = lambda2 = 0.
    
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
    lambda2 : float
        L2 regularization parameter

    Returns
    -------
    float
        Elastic net loss
    """
    return loss(y, x, beta) + l1_regularization(beta, lambda1) + l2_regularization(beta, lambda2)

# Solve the elastic net optimization problem for each corrupted data sample
# with all four objective functions: least squares, Lasso, Ridge, and elastic net.
# Fit to a 10th order polynomial with coefficients alpha.

def x_matrix(n_powers, time):
    """Polynomial power matrix (n_data x n_powers + 1)

    Parameters
    ----------
    n_powers : int
        Polynomial order
    time : numpy.ndarray
        Time vector

    Returns
    -------
    numpy.ndarray
        Polynomial power matrix
    """
    return np.vander(time, n_powers + 1, increasing=True)

n_powers = 10  # Polynomial order (can't be greater than 6 or the solver will fail)
n_alpha = n_powers + 1
n_models = 4
alpha_values = np.zeros((n_samples, n_models, n_alpha))
lambda_amp = 20.0
lambda_pairs = lambda_amp * np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]])
x = x_matrix(n_powers, time)  # Polynomial power matrix
for i in range(n_samples):
    for j in range(n_models):
        beta = cp.Variable(n_alpha)
        lambda1, lambda2 = lambda_pairs[j]
        objective = cp.Minimize(elastic_objective(y[i], x, beta, lambda1, lambda2))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)
        alpha_values[i, j] = beta.value

print("Alpha values for the original data fit:")
print(f"Least Squares: {alpha_values[0, 0]}")
print(f"Lasso: {alpha_values[0, 1]}")
print(f"Ridge: {alpha_values[0, 2]}")
print(f"Elastic Net: {alpha_values[0, 3]}")

# Plot the fits for the original data
fig, ax = plt.subplots()
ax.plot(time, temperature, '.', label='Original Data')
for j in range(n_models):
    ax.plot(time, x @ alpha_values[0, j], label=['Least Squares', 'Lasso', 'Ridge', 'Elastic Net'][j])
ax.legend()

# Plot a 3x3 grid of the fits for the corrupted data
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        ax[i, j].plot(time, y[i * 3 + j], '.', label='Corrupted Data')
        for k in range(n_models):
            ax[i, j].plot(time, x @ alpha_values[i * 3 + j, k], label=['Least Squares', 'Lasso', 'Ridge', 'Elastic Net'][k])
            # ax[i, j].set_title(f"Sample {i * 3 + j}")
            if i == 2:
                ax[i, j].set_xlabel('Time')
            if j == 0:
                ax[i, j].set_ylabel('Temperature')
ax[i, j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

# Multi bar chart the standard deviation of the coefficients for each model
fig, ax = plt.subplots()
bar_width = 0.2
bar_positions = np.arange(n_alpha)
for j in range(n_models):
    ax.bar(
        bar_positions + j * bar_width, np.std(alpha_values[:, j], axis=0), 
        bar_width, 
        label=['Least Squares', 'Lasso', 'Ridge', 'Elastic Net'][j]
    )
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels([f'$\\alpha_{{{i}}}$' for i in range(n_alpha)])
ax.set_ylabel('Standard Deviation of Coefficients')
ax.legend()

# Multi bar chart the mean magnitudes of the coefficients for each model
fig, ax = plt.subplots()
bar_width = 0.2
bar_positions = np.arange(n_alpha)
for j in range(n_models):
    ax.bar(bar_positions + j * bar_width, np.mean(np.abs(alpha_values[:, j]), axis=0), bar_width, label=['Least Squares', 'Lasso', 'Ridge', 'Elastic Net'][j])
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels([f'$\\alpha_{{{i}}}$' for i in range(n_alpha)])
ax.set_ylabel('Mean Coefficient Magnitude')
ax.legend()

# Bar chart the loss for each model
loss_values = np.zeros((n_samples, n_models))
for i in range(n_samples):
    for j in range(n_models):
        loss_values[i, j] = loss(y[i], x, alpha_values[i, j]).value
fig, ax = plt.subplots()
bar_positions = np.arange(n_models)
ax.bar(bar_positions, np.mean(loss_values, axis=0))
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Least Squares', 'Lasso', 'Ridge', 'Elastic Net'])
ax.set_ylabel('Mean Loss (MSE)')
plt.show()