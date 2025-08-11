import matplotlib.pyplot as plt
import numpy as np

def plot_points(coords, measurements):
    coords = np.array(coords)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='orange', label='means')
    ax.scatter(measurements[:,0], measurements[:,1], measurements[:,2], color='green', label='measurements')
    x_range = coords[:,0].max() - coords[:,0].min()
    y_range = coords[:,1].max() - coords[:,1].min()
    z_range = coords[:,2].max() - 0  # since you set zmin to 0
    
    ax.set_xlim(coords[:,0].min() - 0.5, coords[:,0].max() + 0.5)
    ax.set_ylim(coords[:,1].min() - 0.5, coords[:,1].max() + 0.5)
    ax.set_zlim(0, coords[:,2].max())
    ax.legend()
    ax.set_box_aspect([x_range, y_range+1, z_range])
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    plt.show()

def plot_state_covariance_evolution_vs_time(t_list, P_list):
    """
    Plot the evolution of the state covariance matrix P vs. time.

    Parameters:
        t_list: list or array of timestamps (must match length of P_list)
        P_list: list of NxN numpy arrays (P matrices)
    """

    t_list = np.array(t_list)
    P_diag = np.array([np.diag(P) for P in P_list])

    n_states = P_diag.shape[1]
    plt.figure(figsize=(12, 5))
    labels = [f'P[{i},{i}]' for i in range(n_states)]

    for i in range(n_states):
        plt.plot(t_list, P_diag[:, i], label=labels[i])

    plt.title('State Covariance (P) - Diagonal vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()