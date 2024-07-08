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
# Start with the A and B matrices:

#%%
def get_AB(params):
    """Linearized rotary inverted pendulum system dynamics
    
    From https://doi.org/10.1155/2011/528341, equations (35) and (36).
    State vector x = [theta1, theta2, theta1_dot, theta2_dot].
    Operating point is at the upright equilibrium: x = [0, pi, 0, 0].
    Input vector u = [tau1, tau2].
    """
    # Unpack parameters
    m1, m2, l1, l2, L1, L2, J0h, J2h, b1, b2, g = unpack_params(params)

    # Linearized system dynamics
    den = J0h*J2h - m2**2*L1**2*l2**2
    print(den)
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
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [A31, A32, A33, A34],
        [A41, A42, A43, A44]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [B31, B32],
        [B41, B42]
    ])
    return A, B

#%% [markdown]
# ## Full-State Feedback LQR Control
# 
# Define the LQR controller using the control library.
# Begin by defining the Q and R cost function matrices:

#%%
Q = np.diag([
    1,  # theta1 error cost
    1,  # theta2 error cost
    1,  # theta1 rate error cost
    1,  # theta2 rate error cost
])  # State error cost matrix
R = np.diag([
    1,  # tau1 cost
    10000,  # tau2 cost
])  # Control effort cost matrix

# [markdown]
# Next, compute the LQR gain matrix:

#%%
A, B = get_AB(params=params)
sys_lin = control.ss(A, B, np.eye(4), np.zeros((4, 2)))
K, S, E = control.lqr(sys_lin, Q, R)

#% [markdown]
# Define the control law function:

#%%
def lqr_control(x, x_command, K):
    """LQR full-state feedback control law
    
    Incorporate the command state x_command to compute the control input.
    """
    u = -K @ (x - x_command)
    return u

#%% [markdown]
# ## Simulation
#
# Now simulate the nonlinear system with the LQR controller.
# Define the simulation function using scipy for integration:

#%%
def simulate_nonlinear_system(x0, t_sim, x_command, params, ufun):
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
x_command = np.array([np.pi/3, np.pi, 0, 0])  # Command state
def ufun(t, x):  # Control input function
    return lqr_control(x, x_command, K)

#%% [markdown]
# Simulate the response:

#%%
x_sim = simulate_nonlinear_system(x0, t_sim, x_command, params, ufun)

#%% [markdown]
# ## Plot the Closed-Loop Response
#
# Plot the response of the system states and control inputs:

#%%
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax[0].plot(t_sim, x_sim.y[0], label=r'$\theta_1$')
ax[0].plot(t_sim, x_sim.y[1], label=r'$\theta_2$')
ax[0].plot(t_sim, x_command[0] * np.ones_like(t_sim), 'k--', label=r'$\theta_1$ command')
ax[0].plot(t_sim, x_command[1] * np.ones_like(t_sim), 'k:', label=r'$\theta_2$ command')
ax[0].set_ylabel('Angle (rad)')
ax[0].legend()
ax[1].plot(t_sim, x_sim.y[2], label=r'$\dot{\theta}_1$')
ax[1].plot(t_sim, x_sim.y[3], label=r'$\dot{\theta}_2$')
ax[1].plot(t_sim, x_command[2] * np.ones_like(t_sim), 'k--', label=r'$\dot{\theta}_1$ command')
ax[1].plot(t_sim, x_command[3] * np.ones_like(t_sim), 'k:', label=r'$\dot{\theta}_2$ command')
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

def animate_rotary_pendulum(t, x, params):
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
    rod1, = ax.plot([], [], 'r', lw=2)
    rod2, = ax.plot([], [], 'b', lw=2)

    # Update function for the animation
    def update(i):
        theta1 = x[0, i]
        theta2 = x[1, i]
        x0 = 0
        y0 = 0
        x1 = L1 * np.sin(theta1)
        y1 = -1
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        rod1.set_data([x0, x1], [y0, y1])
        rod2.set_data([x1, x2], [y1, y2])
        return rod1, rod2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(len(t)), blit=True, interval=6)
    return anim


#%% [markdown]
# Animate the response:

#%%
anim = animate_rotary_pendulum(t_sim, x_sim.y, params)
plt.show()