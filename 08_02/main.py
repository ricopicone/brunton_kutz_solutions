#%% [markdown]
# First, import the necessary libraries:

#%%
import numpy as np
import matplotlib.pyplot as plt
import control
from scipy import integrate

#%% [markdown]
# Next, define the system parameters:

#%%
params = {
    'm1': 1.0,  # kg
    'm2': 1.0,  # kg
    'l1': 0.5,  # m
    'l2': 0.5,  # m
    'L1': 1.0,  # m
    'L2': 1.0,  # m
    'J0h': 1.0,  # kg*m^2
    'J2h': 1.0,  # kg*m^2
    'b1': 0.1,  # N*m*s/rad
    'b2': 0.1,  # N*m*s/rad
    'g': 9.81  # m/s^2
}

def unpack_params(params):
    """Unpack parameters dictionary into individual variables"""

    m1 = params['m1']
    m2 = params['m2']
    l1 = params['l1']
    l2 = params['l2']
    L1 = params['L1']
    L2 = params['L2']
    J0h = params['J0h']
    J2h = params['J2h']
    b1 = params['b1']
    b2 = params['b2']
    g = params['g']
    return m1, m2, l1, l2, L1, L2, J0h, J2h, b1, b2, g

#%% [markdown]
# ## Define the System Dynamics
#
# Deriving the system dynamics of this system is rather involved and would distract from the main purpose of this exercise.
# We will use the system dynamics provided in the paper _On the Dynamics of the Furuta Pendulum_ by Cazzolato and Prime.
# 
# The nonlinear system dynamics are given by equations (33) and (34).
# See figure 1 in the paper for the system diagram.
# We will assume that the center of mass of each arm is at the geometric center of the arm.
# Let the state vector be
# $$
# x = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \dot{\theta}_1 \\ \dot{\theta}_2 \end{bmatrix},
# $$
# Define the nonlinear system dynamics:

#%%
def rotary_inverted_pendulum(t, x, ufun, params):
    """Rotary inverted pendulum system dynamics
    
    From https://doi.org/10.1155/2011/528341, equations (33) and (34).
    State vector x = [theta1, theta2, theta1_dot, theta2_dot].
    """

    # Unpack parameters
    m1, m2, l1, l2, L1, L2, J0h, J2h, b1, b2, g = unpack_params(params)

    # Unpack states
    theta1, theta2, theta1_dot, theta2_dot = x

    # Unpack inputs
    tau1, tau2 = ufun(t, x)

    # Compute state derivatives and return
    theta_terms = np.array([[
            theta1_dot,
            theta2_dot,
            theta1_dot * theta2_dot,
            theta1_dot**2,
            theta2_dot**2
        ]]).T  # 5x1 matrix
    forcing_terms = np.array([[
            tau1,
            tau2,
            g
        ]]).T  # 3x1 matrix
    constant_factor = 1 / (
        J0h*J2h 
        + J2h**2 * np.sin(theta2)**2 
        - m2**2 * L1**2 * l2**2 * np.cos(theta2)**2
    )
    dtheta1_dt = theta1_dot
    dtheta1_dot_dt = constant_factor  * (
        np.array([[
            -J2h*b1,
            m2*L1*l2 * np.cos(theta2) * b2,
            -J2h**2 * np.sin(2*theta2),
            -1/2 * J2h*m2*L1*l2 * np.cos(theta2) * np.sin(2*theta2),
            J2h*m2*L1*l2 * np.sin(theta2)
        ]]) @ theta_terms +
        np.array([[
            J2h,
            -m2*L1*l2*np.cos(theta2),
            1/2 * m2**2 * l2**2 * L1 * np.sin(2*theta2)
        ]]) @ forcing_terms
    )
    dtheta2_dt = theta2_dot
    dtheta2_dot_dt = constant_factor * (
        np.array([[
            m2*L1*l2 * np.cos(theta2) * b1,
            -b2 * (J0h + J2h * np.sin(theta2)**2),
            m2*L1*l2*J2h * np.cos(theta2) * np.sin(2*theta2),
            -1/2 * np.sin(2*theta2) * (J0h*J2h + J2h**2 * np.sin(theta2)**2),
            -1/2 * m2*2*L1**2*l2 * np.sin(2*theta2)
        ]]) @ theta_terms +
        np.array([[
            -m2*L1*l2 * np.cos(theta2),
            J0h + J2h * np.sin(theta2)**2,
            -m2*l2*np.sin(theta2) * (J0h + J2h * np.sin(theta2)**2)
        ]]) @ forcing_terms
    )
    dx_dt = np.array([
        dtheta1_dt,
        dtheta2_dt,
        dtheta1_dot_dt.item(),
        dtheta2_dot_dt.item()
    ])
    return dx_dt

#%% [markdown]
# Define the linearized system dynamics.
# The paper provides the linearized system dynamics in equations (35) and (36) for the unstable upright equilibrium point
# $$
# x_e = \begin{bmatrix} 0 \\ \pi \\ 0 \\ 0 \end{bmatrix}.
# $$
# It includes torque inputs $\tau_1$ and $\tau_2$, but we will only consider $\tau_1$ in accordance with the problem statement.
# This equilibrium point can be derived by setting the nonlinear system dynamics to zero and solving for the state variables.

#%%
def get_AB(params, tau1only=True, upright=True):
    """Linearized rotary inverted pendulum system dynamics
    
    From https://doi.org/10.1155/2011/528341, equations (35) and (36).
    State vector x = [theta1, theta2, theta1_dot, theta2_dot].
    Operating point is at the upright equilibrium (default) x = [0, pi, 0, 0]
    or the unstable equilibrium x = [0, 0, 0, 0] if upright=False.
    Input vector u = [tau1, tau2] or [tau1] if tau1only=True.
    """
    # Unpack parameters
    m1, m2, l1, l2, L1, L2, J0h, J2h, b1, b2, g = unpack_params(params)

    # Linearized system dynamics
    den = J0h*J2h - m2**2*L1**2*l2**2
    A31 = 0
    A32 = g * m2**2 * l2**2 * L1 / den
    A33 = -b1 * J2h / den
    A34 = -b2 * m2 * l2 * L1 / den
    A41 = 0
    A42 = g * m2 * l2 * J0h / den
    A43 = -b1 * m2 * l2 * L1 / den
    A44 = -b2 * J0h / den
    B31 = J2h / den
    B41 = m2 * L1 * l2 / den
    B32 = m2 * L1 * l2 / den
    B42 = J0h / den
    if upright:
        sign = 1
    else:
        sign = -1
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [A31, A32, A33, sign*A34],
        [A41, sign*A42, sign*A43, A44]
    ])
    if tau1only:  # Single input
        B = np.array([
            [0],
            [0],
            [B31],
            [sign*B41]
        ])
    else:
        B = np.array([
            [0, 0],
            [0, 0],
            [B31, sign*B32],
            [sign*B41, B42]
        ])
    return A, B

#%% [markdown]
# ## Controllability
#
# Check the controllability of the linearized system.
# The system is controllable if the controllability matrix has full rank.
# The controllability matrix is given by

#%%
A, B = get_AB(params, tau1only=True)
Ctrb = control.ctrb(A, B)
rank_Ctrb = np.linalg.matrix_rank(Ctrb)
full_rank = rank_Ctrb == A.shape[0]
print(f'Controllability matrix rank: {rank_Ctrb}')
print(f'Full rank: {full_rank}')

#%% [markdown]
# Since the controllability matrix has full rank, the system is controllable.
# It would be more controllable if both torque inputs $\tau_1$ and $\tau_2$ were used, but it is remarkable that the system is controllable with only one input.
#
# ## Full-State Feedback LQR Control
# 
# Define the LQR controller using the control library.
# Begin by defining the Q and R cost function matrices:

#%%
Q = np.diag([
    1,  # theta1 error cost
    10,  # theta2 error cost
    1,  # theta1 rate error cost
    1,  # theta2 rate error cost
])  # State error cost matrix
R = np.diag([
    1,  # tau1 cost
])  # Control effort cost matrix

#%% [markdown]
# Next, compute the LQR gain matrix:

#%%
A, B = get_AB(params=params, tau1only=True)
sys_lin = control.ss(A, B, np.eye(4), np.zeros((A.shape[0], B.shape[1])))
Kr, S, E = control.lqr(sys_lin, Q, R)

#%% [markdown]
# Define the control law function:

#%%
def lqr_control(x, x_command, Kr):
    """LQR full-state feedback control law
    
    Incorporate the command state x_command to compute the control input.
    """
    u = -Kr @ (x - x_command)
    return u

#%% [markdown]
# ## Simulation
#
# Now simulate the nonlinear system with the LQR controller.
# Define the simulation function using scipy for integration:

#%%
def simulate_nonlinear_system(x0, t_sim, params, ufun):
    """Simulate the nonlinear rotary inverted pendulum system
    
    Use the scipy `solve_ivp` function to simulate the system dynamics.
    """
    x_sim = integrate.solve_ivp(
        rotary_inverted_pendulum,
        t_span=(t_sim[0], t_sim[-1]),
        y0=x0,
        t_eval=t_sim,
        args=(ufun, params),
    )
    return x_sim

#%% [markdown]
# Define the simulation parameters:

#%%
t_sim = np.linspace(0, 10, 1000)  # Simulation time
x0 = np.array([-40, 199, 0, 0]) * np.pi/180  # Initial state
u0 = np.array([0, 0])  # Initial control input
def ufun(t, x, tau1only=True):  # Control input function
    ufun.x_command = np.array([np.pi/3, np.pi, 0, 0])  # Command state (static)
    u =  lqr_control(x, ufun.x_command, Kr)
    if tau1only:
        tau2 = np.zeros_like(u)  # Zero out tau2
        return np.hstack((u, tau2))
    else:
        return u

#%% [markdown]
# Simulate the response:

#%%
x_sim = simulate_nonlinear_system(x0, t_sim, params, ufun)

#%% [markdown]
# ## Plot the Closed-Loop Response
#
# Plot the response of the system states and control inputs:

#%%
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax[0].plot(t_sim, x_sim.y[0], label=r'$\theta_1$')
ax[0].plot(t_sim, x_sim.y[1], label=r'$\theta_2$')
ax[0].plot(t_sim, ufun.x_command[0] * np.ones_like(t_sim), 'k--', label=r'$\theta_1$ command')
ax[0].plot(t_sim, ufun.x_command[1] * np.ones_like(t_sim), 'k:', label=r'$\theta_2$ command')
ax[0].set_ylabel('Angle (rad)')
ax[0].legend()
ax[1].plot(t_sim, x_sim.y[2], label=r'$\dot{\theta}_1$')
ax[1].plot(t_sim, x_sim.y[3], label=r'$\dot{\theta}_2$')
ax[1].plot(t_sim, ufun.x_command[2] * np.ones_like(t_sim), 'k--', label=r'$\dot{\theta}_1$ command')
ax[1].plot(t_sim, ufun.x_command[3] * np.ones_like(t_sim), 'k:', label=r'$\dot{\theta}_2$ command')
ax[1].set_ylabel('Angular Rate (rad/s)')
ax[1].set_xlabel('Time (s)')
plt.draw()

#%% [markdown]
# The response shows that the system is able to stabilize around the commanded state.
#
# Visualize the rotary inverted pendulum response as an animation:

#%%
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def animate_rotary_pendulum(t, x, params, track_theta2=False):
    """Animate the rotary inverted pendulum response"""

    # Unpack parameters
    m1, m2, l1, l2, L1, L2, J0h, J2h, b1, b2, g = unpack_params(params)

    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Initialize the plot elements
    rod01, = ax.plot([], [], 'k--', lw=1)
    rod02, = ax.plot([], [], 'k--', lw=1)
    rod1, = ax.plot([], [], 'r', lw=2)
    rod2, = ax.plot([], [], 'b', lw=2)
    text_theta1 = ax.annotate(
        text=r'$\theta_1 = 0$', 
        xy=(0, -L1/2), xycoords='data', 
        xytext=(20, -20), textcoords='offset pixels', 
        ha='center', va='center'
    )
    text_theta2 = ax.annotate(
        text=r'$\theta_2 = 0$', 
        xy=(0, -L2/2), xycoords='data', 
        xytext=(20, -20), textcoords='offset pixels', 
        ha='center', va='center'
    )

    # Update function for the animation
    def update(i):
        theta1 = x[0, i]
        theta2 = x[1, i]
        x0 = 0
        y0 = 0
        if track_theta2:
            x_theta1 = -L1 * np.sin(theta1)
            y_theta1 = -L1/2 * np.cos(theta1)
            y0d = -L1/2
            x1 = L1 * np.sin(theta1)
            y1 = -L1/2 * np.cos(theta1)
            x2 = x1 + L2 * np.sin(theta2) - x1
            y2 = -L2 * np.cos(theta2) + y0d
            rod01.set_data([x0, -x1], [y0, y1])
            rod02.set_data([x0, x0], [y0d, y0d+L2])
            rod1.set_data([x0, x0], [y0, y0d])
            rod2.set_data([x0, x2], [y0d, y2])
            text_theta1.xy = (x_theta1, y_theta1)
            text_theta1.set_position((-45*np.sin(theta1), -45*np.cos(theta1)))
            text_theta2.xy = (x0, y0d+L2)
            text_theta2.set_position((x0, y0+L2+20))
            return rod01, rod02, rod1, rod2, text_theta1, text_theta2
        else:
            y_theta1 = -L1/2
            x1 = L1 * np.sin(theta1)
            y1 = -L1/2 * np.cos(theta1)
            x2 = x1 + L2 * np.sin(theta2)
            y2 = y1 - L2 * np.cos(theta2)
            rod01.set_data([x0, x0], [y0, y_theta1])
            rod02.set_data([x1, x1], [y1, y2])
            rod1.set_data([x0, x1], [y0, y1])
            rod2.set_data([x1, x2], [y1, y2])
            text_theta1.xy = (x0, y_theta1)
            text_theta1.set_position((0, -20))
            text_theta2.xy = (x1, y2)
            text_theta2.set_position((x1, y2+20))
            return rod01, rod02, rod1, rod2, text_theta1, text_theta2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=6)
    return anim


#%% [markdown]
# Animate the response:

#%%
anim = animate_rotary_pendulum(t_sim, x_sim.y, params, track_theta2=False)
plt.draw()

#%% [markdown]
# Now plot the control inputs over time along with the angular position states:

#%%
u_sim = np.array([ufun(t, x) for t, x in zip(t_sim, x_sim.y.T)])
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax[0].plot(t_sim, x_sim.y[0], label=r'$\theta_1$')
ax[0].plot(t_sim, x_sim.y[1], label=r'$\theta_2$')
ax[0].plot(t_sim, ufun.x_command[0] * np.ones_like(t_sim), 'k--', label=r'$\theta_1$ command')
ax[0].plot(t_sim, ufun.x_command[1] * np.ones_like(t_sim), 'k:', label=r'$\theta_2$ command')
ax[0].set_ylabel('Angle (rad)')
ax[0].legend()
ax[1].plot(t_sim, u_sim[:,0], label=r'$\tau_1$')
ax[1].set_ylabel(r'Control Input $\tau_1$ (N*m)')
ax[1].set_xlabel('Time (s)')
plt.draw()

#%% [markdown]
# ## Observability
#
# Explore the observability of the system.
# We define three different sensor configurations:
# 1. Observe $\theta_1$ and $\theta_2$
# 2. Observe $\dot{\theta}_1$ and $\dot{\theta}_2$
# 3. Observe $\theta_1$ and $\dot{\theta}_1$
#
# Define the output equation $C$ matrix for each sensor configuration:

#%%
def get_C(params, sensor_config):
    """C matrix for the rotary inverted pendulum system
    
    Define the C matrix for different sensor configurations.
    """
    if sensor_config == 1:
        C = np.array([
            [1, 0, 0, 0],  # theta1
            [0, 1, 0, 0]  # theta2
        ])
    elif sensor_config == 2:
        C = np.array([
            [1, 0, 0, 0],  # theta1
            [0, 0, 0, 1]  # theta2_dot
        ])
    elif sensor_config == 3:
        C = np.array([
            [0, 0, 1, 0],  # theta1_dot
            [0, 0, 0, 1]  # theta2_dot
        ])
    return C

#%% [markdown]
# Define the observability matrix for each sensor configuration:

#%%
sensor_configs = [1, 2, 3]
for sensor_config in sensor_configs:
    C = get_C(params, sensor_config)
    Obsv = control.obsv(A, C)
    rank_Obsv = np.linalg.matrix_rank(Obsv)
    full_rank = rank_Obsv == A.shape[0]
    print(f'Sensor Configuration {sensor_config}')
    print(f'\tObservability matrix rank: {rank_Obsv}')
    print(f'\tFull rank: {full_rank}')

#%% [markdown]
# So the system is observable for sensor configurations 1 and 2 only, but not for 3.
# Now explore which of sensor configurations 1 and 2 is more observable.
# We can compute the observability Gramian for each configuration and compare its eigenvalues.
# The control library function `control.gram()` can compute the observability Gramian.
# However, it can only do so for stable systems.
# This is clearly out for the unstable equilibrium.
# The downward equilibrium point $x_e = [0, 0, 0, 0]$ is only marginally stable, so we can't use it, either.
# Let's try nudging the eigenvalues of the downward A matrix to the left, slightly, to make it stable.

#%%
As, Bs = get_AB(params, tau1only=True, upright=False)
print(f'Eigenvalues of A matrix: {np.linalg.eigvals(As)}')
As = As - 0.0001 * np.eye(4)  # Nudge the eigenvalues to the left
print(f'Eigenvalues of nudged A matrix: {np.linalg.eigvals(As)}')
for sensor_config in sensor_configs:
    C = get_C(params, sensor_config)
    sys_lin_down = control.ss(As, Bs, C, np.zeros((C.shape[0], B.shape[1])))
    W = control.gram(sys_lin_down, 'o')
    print(f'Sensor Configuration {sensor_config} (Downward Equilibrium)')
    print(f'\tObservability Gramian Eigenvalues: {np.linalg.eigvals(W)}')
    print(f'\tObservability Gramian Determinant: {np.linalg.det(W):.3e}')

#%% [markdown]
# We see that the observability Gramian for sensor configuration 2 has a higher determinant than for configuration 1.
# Therefore, sensor configuration 2 is more observable than configuration 1.
# However, both configurations are observable.
# We also observe that the observability Gramian for configuration 3 is singular, which is another way of determining that it is not observable for that configuration.
#
# As a practical matter, it is easier to measure angular position than angular velocity.
# Therefore, we will go forward with sensor configuration 1 (observe $\theta_1$ and $\theta_2) for the observer design.
#
# ## Full-State Observer Design
#
# Define the observer gain matrix using the control library.
# Larger values in the observer weighting matrices $V_d$ and $V_n$ will make the observer more sensitive to disturbances (i.e., process noise) and measurement noise, respectively.
# The linear state equation assumed for the observer design is
# \begin{align*}
# \dot{x} &= A x + B u + G w_d \\
# y &= C x + D u + w_n
# \end{align*}
# where $w_d$ is the disturbance input, $G$ is the disturbance input matrix, and $w_n$ is the measurement noise.
# The observer dynamics are given by
# \begin{align*}
# \dot{\hat{x}} &= A \hat{x} + B u + K_f (y - C \hat{x} - D u) \\
# &= (A - K_f C) \hat{x} + (B - K_f D) u + K_f y \\
# &= (A - K_f C) \hat{x} + \begin{bmatrix} K_f & (B - K_f D) \end{bmatrix} \begin{bmatrix} y \\ u \end{bmatrix},
# \end{align*}
# where $\hat{x}$ is the state estimate and $K_f$ is the observer gain matrix.
# The observer gain matrix $K_f$ can be computed using the control library function `control.lqe()` as follows:

#%%
A, B = get_AB(params, tau1only=True)
C = get_C(params, sensor_config=1)
n_states = A.shape[0]  # Number of states
n_inputs = B.shape[1]  # Number of control inputs
n_outputs = C.shape[0]  # Number of outputs
D = np.zeros((n_outputs, n_inputs))
n_disturbances = 4  # Number of disturbance inputs
n_noises = 2  # Number of measurement noise inputs
stdd = 1e-3  # Standard deviation of disturbances
stdn = 1e-2  # Standard deviation of measurement noise
Vd = stdd**2 * np.diag(np.ones((n_disturbances,)))  # Disturbance covar
Vn = stdn**2 * np.diag(np.ones((n_noises,)))  # Measurement noise covar
G = np.eye(A.shape[0])  # Disturbance input matrix
Kf, P, E = control.lqe(A, G, C, Vd, Vn)
sys_lin = control.ss(A, B, C, D)
estim = control.create_estimator_iosystem(
    sys_lin, Vd, Vn, G=G,
    inputs=['theta1_dn', 'theta2_dn', 
        'tau1', 'tau2', 'w_d1', 'w_d2', 'w_d3', 'w_d4', 'w_n1', 'w_n2'],
    outputs=['theta1_hat', 'theta2_hat', 'theta1_dot_hat', 'theta2_dot_hat'],
    name='estim'
)
print(f'Observer Gain Matrix: {Kf}')
print(estim)

#%% [markdown]
# ## LQG Control
#
# Define the LQG controller function.
# The LQG controller combines the LQR controller with the LQE observer.
# The control input is computed as
# $$
# u = -K_f \hat{x}.
# $$
# The observer dynamics have already been defined, above.
# The LQG controller, then, is a dynamic system with input $u$, internal state $\hat{x}$, and output $u$.
#
# When the plant is linear, the closed-loop system can be represented as a single state-space system.
# In our case, the plant is nonlinear, so we will need to model the LQG controller as a separate system, which will be used to compute the control input $u$ at each time step.
# Perhaps the easiest way to do this is to use the control library to create a state-space system for the plant and the LQG controller, then connect them with the `control.interconnect()` function.
# Begin with defining the LQG controller, which has state $\hat{x}$.
# The input is normally the noisy output of the plant $y$, but we will augment it with the command 4-vector $r$, which specifies the desired states.

#%%


#%% [markdown]
# Now create a `control.InputOutputSystem` object for the nonlinear plant.
# Before we can do that, we need to define a plant function that can incorporate disturbances.
# Consider the following:

#%%
def rotary_inverted_pendulum_disturbed(t, x, v, params):
    """Rotary inverted pendulum system dynamics with disturbances
    
    Input vector v = [tau1, tau2, w_d1, w_d2, w_d3, w_d4, w_n1, w_n2].
    State vector x = [theta1, theta2, theta1_dot, theta2_dot].
    """

    n_inputs = 2  # Number of control inputs (tau1, tau2)
    n_disturbances = 4  # Number of disturbance inputs
    # Get the undisturbed state derivatives
    def ufun(t, x):
        return v[:n_inputs]
    dx_dt = rotary_inverted_pendulum(t, x, ufun, params)

    # Add the disturbance terms
    G = np.eye(x.shape[0])  # Disturbance input matrix
        # Each disturbance input affects the corresponding state
    dx_dt += G @ v[n_inputs:n_inputs+n_disturbances]

    return dx_dt


#%% [markdown]
# Additionally, we need a corresponding output function to incorporate measurement noise.
# Consider the following:

#%%
def output_function_noised(t, x, v, params):
    """Output function for the rotary inverted pendulum system
    
    Output function for the rotary inverted pendulum system with measurement noise.
    """
    n_noises = 2  # Number of measurement noise inputs
    y = np.array([x[0], x[1]])  # Output is [theta1_dn, theta2_dn]
    y += v[-n_noises:]  # Add the measurement noise
    return y

#%% [markdown]
# Now create the `control.InputOutputSystem` object for the nonlinear plant:

#%%
sys_plant = control.NonlinearIOSystem(
    rotary_inverted_pendulum_disturbed,
    outfcn=output_function_noised,
    inputs=['tau1', 'tau2', 'w_d1', 'w_d2', 'w_d3', 'w_d4', 'w_n1', 'w_n2'],
    states=['theta1', 'theta2', 'theta1_dot', 'theta2_dot'],
    outputs=['theta1_dn', 'theta2_dn'],
    name='sys_plant',
    params=params
)

# Kr_stack = np.vstack([Kr, np.zeros((7, Kr.shape[1]))])
Kr_stack = np.vstack([Kr, np.zeros((1, Kr.shape[1]))])

print(f"Kr: {Kr}\nKr shape: {Kr.shape}")

ctrl, clsys = control.create_statefbk_iosystem(
    sys_plant, Kr_stack, estimator=estim,
    inputs=[
        'theta1_command', 'theta2_command',  # Command inputs
        'theta1_dot_command', 'theta2_dot_command', 
        'w_d1', 'w_d2', 'w_d3', 'w_d4',  # Disturbance inputs
        'w_n1', 'w_n2'  # Measurement noise inputs
    ],  # External inputs
    # outputs=['theta1_dn', 'theta2_dn', 'tau1', 'tau2'],
    name='clsys'
)

print(clsys)

exit(0)

#%% [markdown]
# Now we can connect the LQG controller to the nonlinear plant using the `control.interconnect()` function as follows:

#%%
sys_cl = control.interconnect(
    syslist=[sys_plant, sysc],
    # connections=[],  # Internal connections are connected by name
    inputs=[
        'theta1_command', 'theta2_command',  # Command inputs
        'theta1_dot_command', 'theta2_dot_command', 
        'w_d1', 'w_d2', 'w_d3', 'w_d4',  # Disturbance inputs
        'w_n1', 'w_n2'  # Measurement noise inputs
    ],  # External inputs
    # inplist=[],  # External inputs are connected by name
    outputs=['theta1_dn', 'theta2_dn', 'tau1', 'tau2'],
    # outlist=[]  # External outputs are connected by name
)

#%% [markdown]
# ## Simulation with LQG Control
#
# Define the simulation function for the LQG-controlled system using the `control.forced_response()` function:

#%%
t_sim = np.linspace(0, .1, 1000)  # Simulation time
x0 = np.array([0, np.pi, 0, 0])  # Initial state (use for observer too)
command_inputs = np.vstack([
    0.1*np.ones_like(t_sim),  # theta1 command
    np.pi*np.ones_like(t_sim),  # theta2 command
    np.zeros_like(t_sim),  # theta1_dot command
    np.zeros_like(t_sim)  # theta2_dot command
])  # Command inputs
# disturbance_inputs = stdd*np.random.randn(4, len(t_sim))  # Disturbances
# noise_inputs = stdn*np.random.randn(2, len(t_sim))  # Measurement noise
disturbance_inputs = 0*np.random.randn(4, len(t_sim))  # Disturbances
noise_inputs = 0*np.random.randn(2, len(t_sim))  # Measurement noise
all_inputs = np.vstack([
    command_inputs, disturbance_inputs, noise_inputs
])  # All inputs
print(all_inputs.shape)
lqr_response = control.forced_response(
    sys_cl,
    T=t_sim,
    U=all_inputs,
    X0=np.concatenate([x0,x0])
)
y = lqr_response.outputs
x = lqr_response.states

#%% [markdown]
# Plot the response of the system states and control inputs:

#%%
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax[0].plot(t_sim, y[0], label=r'$\theta_1$')
ax[0].plot(t_sim, y[1], label=r'$\theta_2$')
ax[0].plot(t_sim, command_inputs[0], 'k--', label=r'$\theta_1$ command')
ax[0].plot(t_sim, command_inputs[1], 'k:', label=r'$\theta_2$ command')
ax[0].set_ylabel('Angle (rad)')
ax[0].legend()
ax[1].plot(t_sim, y[2], label=r'$\tau_1$')
ax[1].plot(t_sim, y[3], label=r'$\tau_2$')
ax[1].set_ylabel('Control Input (N*m)')
ax[1].set_xlabel('Time (s)')
ax[1].legend()
plt.draw()

#%% [markdown]
# Plot the plant states and observer states:

#%%
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)
ax[0].plot(t_sim, x[0], label=r'$\theta_1$')
ax[0].plot(t_sim, x[1], label=r'$\theta_2$')
ax[0].plot(t_sim, x[2], label=r'$\dot{\theta}_1$')
ax[0].plot(t_sim, x[3], label=r'$\dot{\theta}_2$')
ax[0].set_ylabel('State')
ax[0].legend()
ax[1].plot(t_sim, x[4], label=r'$\hat{\theta}_1$')
ax[1].plot(t_sim, x[5], label=r'$\hat{\theta}_2$')
ax[1].plot(t_sim, x[6], label=r'$\hat{\dot{\theta}}_1$')
ax[1].plot(t_sim, x[7], label=r'$\hat{\dot{\theta}}_2$')
ax[1].set_ylabel('Observer State')
ax[1].set_xlabel('Time (s)')
ax[1].legend()
plt.draw()

#%% [markdown]
# Animate the response:

#%%
anim = animate_rotary_pendulum(t_sim, y, params, track_theta2=False)
plt.show()
# anim.save('lqg_control.gif', fps=30)