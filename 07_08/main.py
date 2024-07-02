#%% [markdown]
# First, import the necessary libraries:

#%%
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
# Next, define the Markov chain transition matrix:

#%%
P = np.array([
    [0.0, 0.25, 0.25], [0.40, 0.50, 0.25], [0.60, 0.25, 0.50]
])

#%% [markdown]
# ## Long-term distribution
#
# The long-term distribution of the Markov chain is the eigenvector of the transition matrix corresponding to the eigenvalue of 1. We can find this eigenvector using the `np.linalg.eig` function:

#%%
eigenvalues, eigenvectors = np.linalg.eig(P)
idx = np.argmin(np.abs(eigenvalues - 1.0))  # Index of e-val close to 1
v = np.real(eigenvectors[:, idx])  # Corresponding eigenvector
v = v / np.sum(v)  # Normalize eigenvector
print(f"Long-term distribution: {v}")
print(f"Interpretation:\nSunny probability:{v[0]:.2f}\n"+
      f"Rainy probability: {v[1]:.2f}\n"
      f"Cloudy probability: {v[2]:.2f}")

#%% [markdown]
# ## Simulation
#
# Define an observer function that samples the state of the Markov chain:

#%%
def observer(x):
    """Observe the state of the Markov chain"""
    random_index = np.random.choice(range(3), p=x)
    x = np.zeros(x.shape)
    x[random_index] = 1.0
    return x

#%% [markdown]
# Simulate a random instance (i.e., one corresponding to a random initial condition) of the process, observing the state at each time step:

#%%
T = 100  # Number of time steps
x = np.zeros((T, 3))  # State at each time step
initial_nonzero_index = np.random.randint(0, x.shape[1])
x[0, initial_nonzero_index] = 1.0  # Initial condition
for t in range(0, T-1):
    x_pre_observation = P @ x[t]
    x[t+1] = observer(x_pre_observation)
print(f"First 10 states: {x[:10]}")

#%% [markdown]
# Visualize the simulation. First, convert the state to values that can be plotted:

#%%
x_visualize = np.argmax(x, axis=1)
fig, ax = plt.subplots()
ax.stairs(x_visualize, edges=np.arange(0, len(x_visualize)+1))
ax.set_xlabel("Time step")
ax.set_ylabel("Observed State")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Sunny", "Rainy", "Cloudy"])
ax.set_ylim(-0.1, 2.1)
plt.show()


#%% [markdown]
# # Dynamic Mode Decomposition (DMD)
#
# Define the exact DMD function from Brunton and Kutz (2022):

def DMD(X,Xprime,r):
    # Step 1
    U, Sigma, VT = np.linalg.svd(X, full_matrices=0)
    Ur = U[:, :r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r, :]
    # Step 2
    Atilde = np.linalg.solve(Sigmar.T, (Ur.T @ Xprime @ VTr.T).T).T
    # Step 3
    Lambda, W = np.linalg.eig(Atilde)
    Lambda = np.diag(Lambda)
    # Step 4
    Phi = Xprime @ np.linalg.solve(Sigmar.T, VTr).T @ W
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda, alpha1)
    return Phi, Lambda, b

#%% [markdown]
# Construct the data matrices $X$ and $X'$:

#%%
X = x[:-1].T
Xprime = x[1:].T

#%% [markdown]
# Compute the DMD modes:

#%%
r = 3  # Number of modes
Phi, Lambda, b = DMD(X, Xprime, r)
print(f"Phi:\n{Phi}")
print(f"Lambda:\n{Lambda}")
print(f"b:\n{b}")

#%% [markdown]
# Reconstruct the P matrix from the DMD modes:

#%%
print(f"Phi shape: {Phi.shape}")
print(f"Lambda shape: {Lambda.shape}")
P_dmd = Phi @ Lambda @ np.linalg.inv(Phi)
print(f"P_dmd:\n{P_dmd}")

#%% [markdown]
# Compare the original and reconstructed transition matrices:

#%%
fig, ax = plt.subplots(1, 2)
ax[0].imshow(P, cmap="viridis")
ax[0].set_title("Original P")
ax[1].imshow(P_dmd, cmap="viridis")
ax[1].set_title("Reconstructed P")
plt.show()