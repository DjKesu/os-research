import numpy as np
import matplotlib.pyplot as plt

def bicycle_model(x, u, l):
    """
    Compute the derivative of the state vector.
    
    :param x: State vector [px, py, theta, v, delta]
    :param u: Input vector [a, omega]
    :param l: Wheelbase
    :return: State derivative
    """
    px, py, theta, v, delta = x
    a, omega = u
    
    dxdt = np.array([
        v * np.cos(theta),
        v * np.sin(theta),
        v * np.tan(delta) / l,
        a,
        omega
    ])
    
    return dxdt

def simulate_bicycle(T, dt, l, x0, u_func):
    """
    Simulate the kinematic bicycle model.
    
    :param T: Total simulation time
    :param dt: Time step
    :param l: Wheelbase
    :param x0: Initial state
    :param u_func: Function to compute control input
    :return: Time array and state history
    """
    t = np.arange(0, T, dt)
    x = np.zeros((len(t), 5))
    x[0] = x0
    
    for i in range(1, len(t)):
        u = u_func(t[i])
        k1 = bicycle_model(x[i-1], u, l)
        k2 = bicycle_model(x[i-1] + dt/2 * k1, u, l)
        k3 = bicycle_model(x[i-1] + dt/2 * k2, u, l)
        k4 = bicycle_model(x[i-1] + dt * k3, u, l)
        
        x[i] = x[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, x

# Simulation parameters
T = 10  # Total simulation time
dt = 0.01  # Time step
l = 2  # Wheelbase
x0 = np.array([0, 0, 0, 0, 0])  # Initial state [px, py, theta, v, delta]

# Control input function
def u_func(t):
    return np.array([0.1 * np.cos(t), 0])  # [a, omega]

# Run simulation
t, x = simulate_bicycle(T, dt, l, x0, u_func)

# Plot trajectory
plt.figure(figsize=(10, 6))
plt.plot(x[:, 0], x[:, 1])
plt.title('Vehicle Trajectory')
plt.xlabel('px')
plt.ylabel('py')
plt.grid(True)
plt.axis('equal')
plt.show()