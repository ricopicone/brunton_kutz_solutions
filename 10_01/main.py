#%% [markdown]
# This problem is rather involved, so we will systemaically build up the machinery to solve it.
#
# First, import the necessary libraries:

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import control
import control.optimal as opt
import pydmd
import pysindy
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras import optimizers
import tensorflow as tf

#%% [markdown]
# Set up some flags for running the MPC simulations and test predictions:

#%%
control_lorenz = False
control_DMDc = False
control_SINDYc = False
control_NN = True

retrain = True  # Retrain the NN model

test_DMDc = True
test_SINDYc = True
test_NN = True

#%% [markdown]
# ## Model Predictive Control
#
# We will define an MPC simulation class that can handle nonlinear systems.
# The class will contain a (potentially nonlinear) system model `sys` that is either a plant or a closed-loop system.
# It will also require a predictor function that can predict future states of the plant over a time horizon given the current state and control input.
# It is best for it to be as general as possible, but we won't try to make it capable of handling all possible cases.

#%%
class MPCSimulation:
    """Model Predictive Control Simulation
    
    Simulate a system using model predictive control.

    Attributes:
        sys: System model (plant or closed-loop)
        inplist: List of input variables
        outlist: List of output variables
        predictor: Function that predicts future states given 
            desired current state and control input
        T_horizon: Prediction horizon
        T_update: Update period
        n_updates: Number of updates
        n_horizon: Number of points in prediction horizon
        n_update: Number of points in update period
        xd: Desired state trajectory
        results: Simulation results
    """

    def __init__(self, 
        sys,
        inplist,
        outlist,
        predictor,
        T_horizon, 
        T_update,
        n_updates=10,
        n_horizon=31, 
        n_update=10, 
        xd=None
    ):
        self.sys = sys
        self.inplist = inplist
        self.outlist = outlist
        self.predictor = predictor
        self.T_horizon = T_horizon
        self.T_update = T_update
        self.n_updates = n_updates
        self.n_horizon = n_horizon
        self.n_update = n_update
        self.t_horizon = np.linspace(0, T_horizon, n_horizon)
        self.t_update = np.linspace(0, T_update, n_update+1)
        self.t_sim = np.linspace(0, T_update * n_updates, n_update * n_updates + 1)
        if xd is None:
            xd = np.zeros((sys.nstates, n_update * n_updates + 1))  # Regulate to zero
        self.xd = xd
        self.results = {
            "predictions": {
                "states": np.zeros(
                    (self.sys.nstates, self.n_horizon, self.n_updates)
                ), 
                "inputs": np.zeros(
                    (self.sys.ninputs, self.n_horizon, self.n_updates)
                )
            },
            "simulation": {
                "states": np.zeros(
                    (self.sys.nstates, self.n_update * self.n_updates + 1)
                ), 
                "inputs": np.zeros(
                    (self.sys.ninputs, self.n_update * self.n_updates + 1)
                )
            }
        }  # Store results here
    
    def _predict(self, xd, t_horizon):
        return self.predictor(xd, t_horizon)
    
    def _simulate_update_period(self, xd, period):
        """Simulate over the update period
        
        Implement feedforward control.
        """
        xp, up = self._predict(xd, self.t_horizon)
        self.results["predictions"]["states"][:, :, period] = xp
        self.results["predictions"]["inputs"][:, :, period] = up
        xd = xp[:, :self.n_update+1]
        ud = up[:, :self.n_update+1]
        sim = control.input_output_response(
            self.sys, T=self.t_update, U=ud, X0=xd[:, 0]
        )
        return sim
    
    def simulate(self):
        for i in range(self.n_updates):
            print(f"Simulating update {i+1}/{self.n_updates} ... ", end="", flush=True)
            tic = time.time()
            j = i*self.n_update
            xd_period = self.xd[:, j:j+self.n_update+1]
            if i != 0:
                xd_period[:, 0] = \
                    self.results["simulation"]["states"][:, j]  
                        # Start with last state from previous period
            sim = self._simulate_update_period(xd_period, i)
            toc = time.time()
            print(f"done in {toc-tic:.2f} s.")
            self.results["simulation"]["states"][:, j:j+self.n_update+1] = sim.outputs
            self.results["simulation"]["inputs"][:, j:j+self.n_update+1] = sim.inputs
    
    def plot_results(self, title="MPC Simulation"):
        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.suptitle(title)
        # Plot desired states:
        if self.xd is not None:
            for i in range(self.sys.nstates):
                ax[0].plot(
                    self.t_sim, 
                    self.xd[i, :],
                    'r-.', linewidth=1
                )
        # Plot states:
        ## Plot predicted states:
        for i in range(self.sys.nstates):
            for j in range(self.n_updates):
                ax[0].plot(
                    j*self.T_update, 
                    self.results["predictions"]["states"][i, 0, j],
                    'k.'
                )  # Initial state
                ax[0].plot(
                    self.t_horizon + j*self.T_update, 
                    self.results["predictions"]["states"][i, :, j],
                    'k--', linewidth=0.5
                )  # Predicted state
        ## Plot simulated states:
        for i in range(self.sys.nstates):
            ax[0].plot(
                self.t_sim, 
                self.results["simulation"]["states"][i],
                label=f"State {i}"
            )
        ax[0].set_ylabel('State')
        ax[0].legend()
        # Plot inputs:
        ## Plot predicted inputs:
        for i in range(self.sys.ninputs):
            for j in range(self.n_updates):
                ax[1].plot(
                    j*self.T_update, 
                    self.results["predictions"]["inputs"][i, 0, j],
                    'k.'
                )
                ax[1].plot(
                    self.t_horizon + j*self.T_update, 
                    self.results["predictions"]["inputs"][i, :, j],
                    'k--', linewidth=0.5
                )
        ## Plot simulated inputs:
        for i in range(self.sys.ninputs):
            ax[1].plot(
                self.t_sim, 
                self.results["simulation"]["inputs"][i],
                label=f"Input {i}"
            )
        ax[1].set_ylabel('Input')
        ax[1].set_xlabel('Time')
        ax[1].legend()
        plt.draw()
        return fig, ax
    
    def save(self, filename):
        """Save an MPC simulation object to a pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, filename):
        """Load an MPC simulation object from a pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
            

#%% [markdown]
# The `predictor()` function is quite general here.
# We will write three different versions, one for each of the DMDc, SINDYc, and NN Models.

#%% [markdown]
# ## The Lorenz System and Testing the MPC Simulation
#
# We will use the Lorenz system as the plant model.
# Define the Lorenz system dynamics:

#%%
def lorenz_forced(t, x_, u, params={}):
    """
    Forced Lorenz equations dynamics (dx/dt, dy/dt, dz/dt)
    """
    sigma=10
    beta=8/3
    rho=28
    x, y, z = x_
    dx = sigma * (y - x) + u[0]
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

#%% [markdown]
# Because we're using the `control` package, we can use the `input_output_response()` function to simulate the forced Lorenz system.
# Create a `NonlinearIOSystem` object for the forced Lorenz system:

#%%
lorenz_forced_sys = control.NonlinearIOSystem(
    lorenz_forced, None, inputs=["u"], states=["x", "y", "z"],
    name="lorenz_forced_sys"
)

#%% [markdown]
# We would like to predict a trajectory, state and input, over a time horizon.
# For all three models (and the exact model used here), this will involve predicting the future states given the desired state trajectory.
# The challenge is that we don't know the future inputs.
# There are multiple ways to approach this.
# The first is to assume that the future inputs are the same as the current input.
# This works for short update periods relative to the dynamics of the system.
# The second approach is to solve an optimal control problem to determine the future inputs.
# This will give better results but is more computationally expensive.
# We will write a function that can handle both cases.

#%%
def predict_trajectory(sys, x0, t_horizon, u0=None, cost=None, terminal_cost=None):
    if cost is None:  # Constant input
        if u0 is None:
            u0 = np.zeros_like(t_horizon)  # Zero input
        u = u0 * np.ones_like(t_horizon)  # Constant input
        x = control.input_output_response(
            sys, T=t_horizon, U=u, x0=x0
        ).states
    else:  # Solve optimal control problem
        ocp = opt.OptimalControlProblem(
            sys, t_horizon, cost, terminal_cost=terminal_cost
        )
        res = ocp.compute_trajectory(x0, print_summary=False)
        u = res.inputs
        x = res.states
    return x, u

#%% [markdown]
# This isn't specific enough to be used as a predictor function, but it can be used to write a predictor function.
# We will write a predictor function that uses the Lorenz system model to predict future states.
# We will use the optimal control approach to determine the future inputs.

#%%
def lorenz_predictor(xd, t_horizon, sys):
    """Predictor for Lorenz system using optimal control"""
    Q = np.eye(sys.nstates)
    R = 0.01 * np.eye(sys.ninputs)
    cost = control.optimal.quadratic_cost(sys, Q, R, x0=xd[:, -1])
    terminal_cost = control.optimal.quadratic_cost(
        sys, 5*Q, 0*R, x0=xd[:, -1]
    )  # Penalize terminal state more
    x, u = predict_trajectory(
        sys, xd[:, 0], t_horizon, cost=cost, terminal_cost=terminal_cost
    )
    return x, u

#%% [markdown]
# We can now test the MPC simulation class using the Lorenz system model to predict future inputs.
# We expect the results to be good because we're using the exact model.

#%%
dt_lorenz = 1e-3  # Time step
T_horizon = dt_lorenz * 50
T_update = dt_lorenz * 20
n_horizon = int(np.floor(T_horizon/dt_lorenz)) + 1
n_update = int(np.floor(T_update/dt_lorenz)) + 1
n_updates = 50
xeq = np.array([-np.sqrt(72), -np.sqrt(72), 27])
print(f"xeq: {xeq}")
command = np.outer(xeq, np.ones(n_update * n_updates + 1))
command[:, 0] = np.array([0, 0, 0])  # Initial state
def _lorenz_predictor(xd, t_horizon):
    return lorenz_predictor(xd, t_horizon, sys=lorenz_forced_sys)
if not control_lorenz:
    mpc_lorenz = MPCSimulation.load("mpc_lorenz.pickle")
else:
    mpc_lorenz = MPCSimulation(
        sys=lorenz_forced_sys,
        inplist=['u'],
        outlist=['lorenz_forced_sys.x', 'lorenz_forced_sys.y', 'lorenz_forced_sys.z'],
        predictor=_lorenz_predictor,
        T_horizon=T_horizon, 
        T_update=T_update,
        n_updates=n_updates,
        n_horizon=n_horizon, 
        n_update=n_update, 
        xd=command
    )
    results_mpc_lorenz = mpc_lorenz.simulate()
    mpc_lorenz.save("mpc_lorenz.pickle")
mpc_lorenz.plot_results("MPC Simulation with Lorenz System")
plt.draw()

#%% [markdown]
# The results are good.
# Note that if a global optimal control problem is solved and used as the `xd` commanded trajectory, the results will be better.
# Here we have used a constant commanded trajectory, which is not a feasible trajectory for the Lorenz system.
# The deviations of the predictor from the actual response are due to the Lorenz system being chaotic, numerical optimization limitations, and the finite time horizon over which the optimal control problem is solved.

# ## Generating Training and Testing Data
#
# Now we generate some training and testing data for the predictors.
# Generate the training and testing data as follows:

#%%
dt_data = 1e-3  # Time step
t_data = np.arange(0, 20, dt_data)  # Time array
n_data = len(t_data)
n_train = int(n_data/2)
u_data_train = (2*np.sin(t_data[:n_train])
                + np.sin(0.1*t_data[:n_train]))**2  # Input
u_data_test = (5*np.sin(30*t_data[n_train:]))**3
u_data = np.hstack((u_data_train, u_data_test))
x_data = control.input_output_response(
    lorenz_forced_sys, T=t_data, U=u_data
).states

#%% [markdown]
# Plot the data over time:

#%%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t_data, x_data[0], label='x')
ax[0].plot(t_data, x_data[1], label='y')
ax[0].plot(t_data, x_data[2], label='z')
ax[0].set_ylabel('State')
ax[0].legend()
ax[1].plot(t_data, u_data, label='u', color='r')
ax[1].set_ylabel('Input')
ax[1].set_xlabel('Time')
plt.draw()

#%% [markdown]
# Partition the data into training and testing sets:

#%%
n_train = int(n_data/2)
n_test = n_data - n_train
t_train = t_data[:n_train]
t_test = t_data[n_train:]
u_train = u_data[:n_train]
u_test = u_data[n_train:]
x_train = x_data[:, :n_train]
x_test = x_data[:, n_train:]

#%% [markdown]
# ## Dynamic Mode Decomposition with Control (DMDc) Model
#
# We could define the exact DMDc function from Brunton and Kutz (2022) section 7.2 and modified based on section 10.2.
# However, it is more convenient to use the PyDMD package to compute the DMDc model.
#
# Compute the DMDc model:

#%%
dmdc = pydmd.DMDc()
dmdc.fit(X=x_train, I=u_train[:-1])
Phi_pydmd = dmdc.modes
Lambda_pydmd = np.diag(dmdc.eigs)
b_pydmd = dmdc.amplitudes
A_pydmd = Phi_pydmd @ Lambda_pydmd @ np.linalg.pinv(Phi_pydmd)
B_pydmd = dmdc.B
print(f"A_pydmd:\n{A_pydmd}")
print(f"B_pydmd:\n{B_pydmd}")

#%% [markdown]
# Predict the trajectory on the test data:

#%%
if test_DMDc:
    dt = t_train[1] - t_train[0]
    x0 = x_test[:, 0]
    sys_DMDc = control.ss(A_pydmd, B_pydmd, np.eye(A_pydmd.shape[0]), 0, dt=dt)
    x_DMDc_pred = control.forced_response(sys_DMDc, T=t_test, U=u_test, X0=x0).states

#%% [markdown]
# Plot the predicted trajectory with the test data:

#%%
if test_DMDc:
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t_test, x_test[0], label='x_test')
    ax[0].plot(t_test, x_DMDc_pred[0], label='x_DMDc_pred')
    ax[0].set_ylabel('x')
    ax[0].legend()
    ax[1].plot(t_test, x_test[1], label='y_test')
    ax[1].plot(t_test, x_DMDc_pred[1], label='y_DMDc_pred')
    ax[1].set_ylabel('y')
    ax[1].legend()
    ax[2].plot(t_test, x_test[2], label='z_test')
    ax[2].plot(t_test, x_DMDc_pred[2], label='z_DMDc_pred')
    ax[2].set_ylabel('z')
    ax[2].set_xlabel('Time')
    ax[2].legend()
    plt.draw()

#%% [markdown]
# The results are so bad because the DMDc model is linear and the Lorenz system is highly nonlinear.
# With an MPC controller, the prediction doesn't have to be good for long, but these predictions deviate almost immediately, so we don't have high hopes for the MPC controller with DMDc.
#
# Nonetheless, define the DMDc predictor function:

#%%
def DMDc_predictor(xd, t_horizon, sys=None):
    """Predictor for DMDc model using optimal control"""
    Q = np.eye(sys.nstates)
    R = 0.01 * np.eye(sys.ninputs)
    cost = control.optimal.quadratic_cost(sys, Q, R, x0=xd[:, -1])
    terminal_cost = control.optimal.quadratic_cost(sys, 5*Q, 0*R, x0=xd[:, -1])
    x, u = predict_trajectory(
        sys, xd[:, 0], t_horizon, cost=cost, terminal_cost=terminal_cost
    )
    return x, u

#%% [markdown]
# Now that we have our DMCc predictor, we can try it in the MPC simulation.
# We could use feedback control as well, which would improve the results, but using feedforward only allows us to get a better comparison among the predictors.
# Again, we don't expect good results, but we can at least see how it performs.

#%%
T_horizon = dt_data * 50
T_update = dt_data * 20
n_horizon = int(np.floor(T_horizon/dt_data)) + 1  # Must match DMDc model timebase
n_update = int(np.floor(T_update/dt_data)) + 1
n_updates = 50
xeq = np.array([-np.sqrt(72), -np.sqrt(72), 27])
print(f"xeq: {xeq}")
command = np.outer(xeq, np.ones(n_update * n_updates + 1))
command[:, 0] = x_test[:, 0]  # Initial state
def DMDc_predictor_(xd, t_horizon):  # Wrapper for predictor (picklable)
    return DMDc_predictor(xd, t_horizon, sys=sys_DMDc)
if not control_DMDc:
    mpc_DMDc = MPCSimulation.load("mpc_DMDc.pickle")
else:
    sys_DMDc = control.ss(A_pydmd, B_pydmd, np.eye(A_pydmd.shape[0]), 0, dt=dt_data)
    mpc_DMDc = MPCSimulation(
        sys=lorenz_forced_sys,
        inplist=['u'],
        outlist=['lorenz_forced_sys.x', 'lorenz_forced_sys.y', 'lorenz_forced_sys.z'],
        predictor=DMDc_predictor_,
        T_horizon=T_horizon, 
        T_update=T_update,
        n_updates=n_updates,
        n_horizon=n_horizon, 
        n_update=n_update, 
        xd=command
    )
    results_mpc_DMDc = mpc_DMDc.simulate()
    mpc_DMDc.save("mpc_DMDc.pickle")
mpc_DMDc.plot_results("MPC Simulation with DMDc Model")
plt.draw()

#%% [markdown]
# As expected, the DMDc model performs poorly.
# An alternative DMDc approach for highly nonlinear systems is to use extended DMDc (EDMDc) with nonlinear measurements.
# This is connected to the Koopman operator theory.
# We will not pursue this here.
#
# ## Sparse Identification of Nonlinear Dynamics (SINDy) 
# 
# Define the SINDy model:

#%%
sindy_model = pysindy.SINDy(feature_names=["x", "y", "z", "u"])
sindy_model.fit(x_train.T, t=dt_data, u=u_train, multiple_trajectories=False)
print("Dynamics identified by pySINDy:")
sindy_model.print()

def extract_sindy_dynamics(sindy_model, eps=1e-12):
    """Extract SINDy dynamics"""
    variables = sindy_model.feature_names  # ["x", "y", "z", "u"]
    coefficients = sindy_model.coefficients()
    features = sindy_model.get_feature_names()  
        # ["1", "x", "y", "z", "u", "x * y", "x * z", "x * u", "y * z", ...]
    features = [f.replace("^", "**") for f in features]
    features = [f.replace(" ", " * ") for f in features]
    def rhs(coefficients, features):
        rhs = []
        for row in range(coefficients.shape[0]):
            rhs_row = ""
            for col in range(coefficients.shape[1]):
                if np.abs(coefficients[row, col]) > eps:
                    if rhs_row:
                        rhs_row += " + "
                    rhs_row += f"{coefficients[row, col]} * {features[col]}"
            rhs.append(rhs_row)
        return rhs
    rhs_str = rhs(coefficients, features)  # Eager evaluation
    n_equations = len(rhs_str)
    def sindy_dynamics(t, x_, u_, params={}):
        states_inputs = x_.tolist() + np.atleast_1d(u_).tolist()
        variables_dict = dict(zip(variables, states_inputs))
        return [eval(rhs_str[i], variables_dict) for i in range(n_equations)]
    return sindy_dynamics

#%% [markdown]
# Let's predict the trajectory on the test data:

#%%
if test_SINDYc:
    sindy_dynamics = extract_sindy_dynamics(sindy_model)
    sindy_sys = control.NonlinearIOSystem(
        sindy_dynamics, None, inputs=["u"], states=["x", "y", "z"],
        name="sindy_sys"
    )
    x_SINDy_pred = control.input_output_response(
        sindy_sys, T=t_test, U=u_test, X0=x_test[:, 0]
    ).states

#%% [markdown]
# Plot the simulated and predicted trajectory with the test data:

#%%
if test_SINDYc:
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t_test, x_test[0], label='x_test')
    ax[0].plot(t_test, x_SINDy_pred[0, :], label='x_SINDy_pred')
    ax[0].set_ylabel('x')
    ax[0].legend()
    ax[1].plot(t_test, x_test[1], label='y_test')
    ax[1].plot(t_test, x_SINDy_pred[1, :], label='y_SINDy_pred')
    ax[1].set_ylabel('y')
    ax[1].legend()
    ax[2].plot(t_test, x_test[2], label='z_test')
    ax[2].plot(t_test, x_SINDy_pred[2, :], label='z_SINDy_pred')
    ax[2].set_ylabel('z')
    ax[2].set_xlabel('Time')
    ax[2].legend()
    plt.draw()

#%% [markdown]
# So the SINDy model is not perfect, but it is much better than the DMDc model.
# For a few seconds, the model is quite good.
# With more training data, the model would likely improve.
#
# We can use the model to write a predictor function for the SINDy model.
# The optimal control approach requires we create a `control.NonlinearIOSystem` object from the SINDy model.

#%%
sindy_dynamics = extract_sindy_dynamics(sindy_model)
sys_SINDy = control.NonlinearIOSystem(
    sindy_dynamics, None, inputs=["u"], states=["x", "y", "z"],
    name="sys_SINDy"
)

#%% [markdown]
# Define the SINDy predictor function:

#%%
def SINDy_predictor(xd, t_horizon, sys=None):
    """Predictor for SINDy model using optimal control"""
    Q = np.eye(sys.nstates)
    R = 0.01 * np.eye(sys.ninputs)
    cost = control.optimal.quadratic_cost(sys, Q, R, x0=xd[:, -1])
    terminal_cost = control.optimal.quadratic_cost(sys, 5*Q, 0*R, x0=xd[:, -1])
    x, u = predict_trajectory(
        sys, xd[:, 0], t_horizon, cost=cost, terminal_cost=terminal_cost
    )
    return x, u

#%% [markdown]
# Now we can test the SINDy model in the MPC simulation.
# We expect the results to be about as good as the exact model.

#%%
T_horizon = dt_data * 50
T_update = dt_data * 20
n_horizon = int(np.floor(T_horizon/dt_data)) + 1
n_update = int(np.floor(T_update/dt_data)) + 1
n_updates = 50
xeq = np.array([-np.sqrt(72), -np.sqrt(72), 27])
print(f"xeq: {xeq}")
command = np.outer(xeq, np.ones(n_update * n_updates + 1))
command[:, 0] = np.array([0, 0, 0])  # Initial state
def SINDy_predictor_(xd, t_horizon):  # Wrapper for predictor (picklable)
    return SINDy_predictor(xd, t_horizon, sys=sys_SINDy)
if not control_SINDYc:
    mpc_SINDYc = MPCSimulation.load("mpc_SINDYc.pickle")
else:
    mpc_SINDYc = MPCSimulation(
        sys=lorenz_forced_sys,
        inplist=['u'],
        outlist=['lorenz_forced_sys.x', 'lorenz_forced_sys.y', 'lorenz_forced_sys.z'],
        predictor=SINDy_predictor_,
        T_horizon=T_horizon, 
        T_update=T_update,
        n_updates=n_updates,
        n_horizon=n_horizon, 
        n_update=n_update, 
        xd=command
    )
    results_mpc_SINDYc = mpc_SINDYc.simulate()
    mpc_SINDYc.save("mpc_SINDYc.pickle")
mpc_SINDYc.plot_results("MPC Simulation with SINDy Model")
plt.draw()

#%% [markdown]
# The results are quite good, very similar to the results with the exact model.
# The SINDy model is a good choice for the MPC simulation.
#
# ## Neural Network (NN) Model
#
# We can train a NN model to predict the Lorenz system dynamics, as in Brunton and Kutz (2022) problem 6.1d.
#
# Begin by defining the neural network architecture:

#%%
def build_model():
    """Build the feedforward neural network model"""
    model = Sequential()
    model.add(Input(shape=(4,)))  # 3 states + 1 input
    model.add(Dense(5))
    model.add(Activation('relu'))
    model.add(Dense(3))  # 3 states
    return model

#%% [markdown]
# Compile the model:

#%%
model = build_model()
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',  # Loss function
    metrics=['mean_absolute_error'],  # Metrics to monitor
)

#%% [markdown]
# Train the model:

#%% tags=["remove_output"]
X = np.hstack([x_train[:, :-1].T, u_train[:-1].reshape(-1, 1)])
Y = x_train[:, 1:].T
if retrain:
    history = model.fit(
        X,  # Input data
        Y,  # Target data
        epochs=30,  # Number of epochs
        batch_size=5,  # Batch size
        validation_split=0.2,  # Validation split
        shuffle=True,  # Shuffle the data
    )
    model.save('model.keras')
    history = True
else:
    model = keras.models.load_model('model.keras')
    history = False

#%% [markdown]
# Plot the training and validation loss versus the epoch:

#%%
if history:
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(model.history.history['loss'], label='Training loss')
    ax.plot(model.history.history['val_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.draw()

#%% [markdown]
# Now we can test the NN model on the test data:

#%%
if test_NN:
    X_test = np.hstack([x_test.T, u_test.reshape(-1, 1)])
    Y_test = x_test.T
    Y_NN_pred = model.predict(X_test).T

#%% [markdown]
# Plot the predicted trajectory with the test data:

#%%
if test_NN:
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(t_test, x_test[0], label='x_test')
    ax[0].plot(t_test, Y_NN_pred[0], label='x_NN_pred')
    ax[0].set_ylabel('x')
    ax[0].legend()
    ax[1].plot(t_test, x_test[1], label='y_test')
    ax[1].plot(t_test, Y_NN_pred[1], label='y_NN_pred')
    ax[1].set_ylabel('y')
    ax[1].legend()
    ax[2].plot(t_test, x_test[2], label='z_test')
    ax[2].plot(t_test, Y_NN_pred[2], label='z_NN_pred')
    ax[2].set_ylabel('z')
    ax[2].set_xlabel('Time')
    ax[2].legend()
    plt.draw()

#%% [markdown]
# The results are quite good.
# There is some deviation, but it doesn't occur until later in the simulation.
# In fact, surprisingly, the NN model is better than the SINDYc model (however, note that the SINDYc model also yields the dynamics, whereas the NN yields only predictions).

# Before we can write a predictor function, we need to create a `control` system object from the NN model.

#%%
@tf.function  # Decorator for TensorFlow function
def NN_predict(X):
    """Predict using the NN model (faster than model.predict)"""
    return model(X)
def NN_dynamics(t, x_, u_, params={}):
    """NN dynamics"""
    X = np.hstack([x_, u_])[np.newaxis, :]
    return NN_predict(X).numpy().flatten()
sys_NN = control.NonlinearIOSystem(
    NN_dynamics, None, inputs=["u"], states=["x", "y", "z"], 
    dt=dt_data, name="sys_NN"
)

t = t_test
y_NN = control.input_output_response(sys_NN, T=t, U=u_test, X0=x_test[:, 0]).outputs
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(t, y_NN[0], label='x')
ax[0].set_ylabel('x')
ax[0].legend()
ax[1].plot(t, y_NN[1], label='y')
ax[1].set_ylabel('y')
ax[1].legend()
ax[2].plot(t, y_NN[2], label='z')
ax[2].set_ylabel('z')
ax[2].set_xlabel('Time')
ax[2].legend()
plt.show()

#%% [markdown]
# Now create a predictor function for the NN model:

#%%
def NN_predictor(xd, t_horizon, sys):
    """Predictor for NN model using optimal control"""
    Q = np.eye(sys.nstates)
    R = 0.01 * np.eye(sys.ninputs)
    cost = control.optimal.quadratic_cost(sys, Q, R, x0=xd[:, -1])
    terminal_cost = control.optimal.quadratic_cost(sys, 5*Q, 0*R, x0=xd[:, -1])
    x, u = predict_trajectory(
        sys, xd[:, 0], t_horizon, cost=cost, terminal_cost=terminal_cost
    )
    return x, u

#%% [markdown]
# Now we can test the NN model in the MPC simulation.

#%%
def NN_predictor_(xd, t_horizon):  # Wrapper for predictor (picklable)
    return NN_predictor(xd, t_horizon, sys=sys_NN)
if not control_NN:
    mpc_NN = MPCSimulation.load("mpc_NN.pickle")
else:
    mpc_NN = MPCSimulation(
        sys=lorenz_forced_sys,
        inplist=['u'],
        outlist=['lorenz_forced_sys.x', 'lorenz_forced_sys.y', 'lorenz_forced_sys.z'],
        predictor=NN_predictor_,
        T_horizon=T_horizon, 
        T_update=T_update,
        n_updates=n_updates,
        n_horizon=n_horizon, 
        n_update=n_update, 
        xd=command
    )
    results_mpc_NN = mpc_NN.simulate()
    mpc_NN.save("mpc_NN.pickle")
mpc_NN.plot_results("MPC Simulation with NN Model")
plt.draw()

#%%
plt.show()
