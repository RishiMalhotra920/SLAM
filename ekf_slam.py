import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 10))

landmarks = np.array([[5, 5], [4, 6], [6, 4]])

true_robot_trajectory = []
# dead_reckoning_trajectory = []
belief_robot_trajectory = []

state_len = 3 + 2 * len(landmarks)
true_state_prev = np.zeros((state_len, 1))
mu_prev = np.zeros((state_len, 1))
sigma_prev = np.eye(state_len)
N_t_prev = 0
np.fill_diagonal(sigma_prev[3:, 3:], 10)


def get_u_t(t):
    """
    u_v and u_omega are control inputs which come with some noise
    v, omega are the true control inputs
    """
    v_gt = 1
    omega_gt = 0.1

    u_v = np.random.normal(v_gt, 0.1)
    u_omega = np.random.normal(omega_gt, 0.01)

    return np.array([v_gt, omega_gt]), np.array([u_v, u_omega])


def get_true_state(state: np.ndarray, u_t: np.ndarray):
    """
    Returns the true state of the robot at time t
    """
    x, y, theta = state[:3]
    v, omega = u_t

    x = x + v * np.cos(theta)
    y = y + v * np.sin(theta)
    theta = theta + omega

    return np.array([x, y, theta] + state[3:])


def get_z_t(true_state: np.ndarray) -> np.ndarray:
    """given true state, return the sensor measurements"""
    raise NotImplementedError


def run_ekf_slam(
    mu_prev: np.ndarray,
    sigma_prev: np.ndarray,
    u_t: np.ndarray,
    z_t: np.ndarray,
    N_t_prev,
) -> np.ndarray:
    raise NotImplementedError


def animate(frame):
    global mu_prev, sigma_prev, true_state_prev, N_t_prev
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # make the state transition

    u_t_gt, u_t_sensor = get_u_t(frame)
    true_state = get_true_state(true_state_prev, u_t_gt)
    z_t = get_z_t(true_state_prev)

    mu, sigma = run_ekf_slam(mu_prev, sigma_prev, u_t_sensor, z_t, N_t_prev)

    belief_robot_trajectory.append(mu[:3])
    true_robot_trajectory.append(true_state[:3])

    mu_prev = mu
    sigma_prev = sigma

    # Draw the true landmarks
    ax.scatter(
        landmarks[:, 0], landmarks[:, 1], c="blue", label="Landmark Ground Truth"
    )

    # Draw the robot
    mu_x = mu[0]
    mu_y = mu[1]
    mu_theta = mu[2]
    arrow_length = 1
    dx = arrow_length * np.cos(mu_theta)
    dy = arrow_length * np.sin(mu_theta)
    ax.arrow(
        mu_x,
        mu_y,
        dx,
        dy,
        head_width=0.1,
        head_length=0.2,
        fc="red",
        ec="red",
        label="Robot",
    )

    # Draw the robot true trajectory
    ax.plot(
        [x[0] for x in true_robot_trajectory],
        [x[1] for x in true_robot_trajectory],
        c="blue",
        label="True Robot Trajectory",
    )

    # Draw the robot belief trajectory
    ax.plot(
        [x[0] for x in belief_robot_trajectory],
        [x[1] for x in belief_robot_trajectory],
        c="green",
        label="True Robot Trajectory",
    )

    # Draw the landmark beliefs
    landmark_beliefs = mu[3:].reshape(-1, 2)
    ax.scatter(
        landmark_beliefs[:, 0],
        landmark_beliefs[:, 1],
        c="gray",
        label="Landmark Beliefs",
    )
    ax.legend()
    return ax


ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=False)

plt.show()
