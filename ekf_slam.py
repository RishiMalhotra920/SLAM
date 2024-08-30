import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 10))

true_landmarks = np.array([[5, 5], [4, 6], [6, 4]])

true_robot_trajectory = []
dead_reckoning_robot_trajectory = []
belief_robot_trajectory = []

state_len = 3 + 2 * len(true_landmarks)
true_state_prev = np.zeros((state_len))
mu_prev = np.zeros(state_len)
sigma_prev = np.eye(state_len)
N_t_prev = 0
R_t = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
np.fill_diagonal(sigma_prev[3:, 3:], 10)

# TODO:
# to not get weird behavior, keep
# angles between -pi and pi.


def get_u_t(t):
    """
    u_v and u_omega are control inputs which come with some noise
    v, omega are the true control inputs

    returns the true control input and the noisy control input
    """
    # TODO: switched around. the control input that should be reported
    # should be the perfect velocity.f
    # the true velocity should be the noisy one
    v_gt = 0.25
    omega_gt = 0.1

    u_v = np.random.normal(v_gt, 0.1)
    u_omega = np.random.normal(omega_gt, 0.01)

    return np.array([v_gt, omega_gt]), np.array([u_v, u_omega])


def get_true_state(state: np.ndarray, u_t: np.ndarray):
    """
    Returns the true state of the robot at time t
    """
    delta_t = 1
    x, y, theta = state[:3]
    v, omega = u_t

    motion_array = np.array(
        [
            -v / omega * np.sin(theta) + v / omega * np.sin(theta + omega * delta_t),
            v / omega * np.cos(theta) - v / omega * np.cos(theta + omega * delta_t),
            omega * delta_t,
        ]
    )

    state_after_motion = state.copy()

    state_after_motion[:3] = state[:3] + motion_array

    return state_after_motion


def get_z_t(true_state: np.ndarray, true_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """given true state, return the sensor measurements"""

    r_t = np.linalg.norm(true_landmarks - true_state[:2], axis=1)
    angle_to_landmark = np.arctan2(true_landmarks[:, 1] - true_state[1], true_landmarks[:, 0] - true_state[0])
    theta_t = normalize_angle(angle_to_landmark - true_state[2])
    z_t = np.array([r_t, theta_t]).T  # shape (n_landmarks, 2)
    z_t_sensor = z_t + np.random.normal(0, 0.1, z_t.shape)

    c_t = np.arange(len(true_landmarks))

    return z_t_sensor, c_t
    # raise NotImplementedError


def run_ekf_slam_known_correspondences(
    mu_prev: np.ndarray, sigma_prev: np.ndarray, u_t: np.ndarray, z_t: np.ndarray, c_t: np.ndarray, R_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # returns mu_bar_t, sigma_bar_t, mu_t, sigma_t

    # maintenance
    v_t, omega_t = u_t
    delta_t = 1

    # step 1
    F_x = np.zeros((3, state_len))
    F_x[:3, :3] = np.eye(3)

    # step 2
    motion_array = np.array(
        [
            -v_t / omega_t * np.sin(mu_prev[2]) + v_t / omega_t * np.sin(mu_prev[2] + omega_t * delta_t),
            v_t / omega_t * np.cos(mu_prev[2]) - v_t / omega_t * np.cos(mu_prev[2] + omega_t * delta_t),
            omega_t * delta_t,
        ]
    )
    mu_bar_t = mu_prev + F_x.T @ motion_array

    # step 3
    jacobian_delta = np.array(
        [
            [0, 0, -v_t / omega_t * np.cos(mu_prev[2]) + v_t / omega_t * np.cos(mu_prev[2] + omega_t * delta_t)],
            [0, 0, -v_t / omega_t * np.sin(mu_prev[2]) + v_t / omega_t * np.sin(mu_prev[2] + omega_t * delta_t)],
            [0, 0, 0],
        ]
    )
    G_t = np.eye(state_len) + F_x.T @ np.array(jacobian_delta) @ F_x

    # step 4
    sigma_bar_t = G_t @ sigma_prev @ G_t.T + F_x.T @ R_t @ F_x

    return mu_bar_t, sigma_bar_t, mu_bar_t, sigma_bar_t


def normalize_angle(theta):
    # keeps angles between -pi and pi
    # kinda wild that this works
    return np.arctan2(np.sin(theta), np.cos(theta))


def animate(frame):
    global mu_prev, sigma_prev, true_state_prev, N_t_prev, true_landmarks, R_t
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # make the state transition

    u_t_gt, u_t_sensor = get_u_t(frame)
    true_state = get_true_state(true_state_prev, u_t_gt)
    z_t, c_t = get_z_t(true_state_prev, true_landmarks)

    mu_bar, sigma_bar, mu_t, sigma_t = run_ekf_slam_known_correspondences(mu_prev, sigma_prev, u_t_sensor, z_t, c_t, R_t)

    true_robot_trajectory.append(true_state[:3])
    dead_reckoning_robot_trajectory.append(mu_bar[:3])
    belief_robot_trajectory.append(mu_t[:3])

    # TODO: change this to mu_t, sigma_t
    true_state_prev = true_state
    mu_prev = mu_bar
    sigma_prev = sigma_bar

    # Draw the true landmarks
    ax.scatter(
        true_landmarks[:, 0],
        true_landmarks[:, 1],
        c="blue",
        label="Landmark Ground Truth",
    )

    # Draw the robot
    arrow_length = 1
    dx = arrow_length * np.cos(true_state[2])
    dy = arrow_length * np.sin(true_state[2])
    ax.arrow(
        true_state[0],
        true_state[1],
        dx,
        dy,
        head_width=0.1,
        head_length=0.2,
        fc="blue",
        ec="blue",
        label="Robot",
    )

    # Draw the robot true trajectory
    ax.plot(
        [x[0] for x in true_robot_trajectory],
        [x[1] for x in true_robot_trajectory],
        c="blue",
        label="True Robot Trajectory",
    )

    # Draw the robot dead reckoning trajectory
    ax.plot(
        [x[0] for x in dead_reckoning_robot_trajectory],
        [x[1] for x in dead_reckoning_robot_trajectory],
        c="black",
        label="Dead reckoning Robot Trajectory",
    )

    # # Draw the robot belief trajectory
    # ax.plot(
    #     [x[0] for x in belief_robot_trajectory],
    #     [x[1] for x in belief_robot_trajectory],
    #     c="green",
    #     label="Robot Belief Trajectory",
    # )

    # Draw the landmark beliefs
    # landmark_beliefs = mu_t[3:].reshape(-1, 2)
    # ax.scatter(
    # landmark_beliefs[:, 0],
    # landmark_beliefs[:, 1],
    # c="gray",
    # label="Landmark Beliefs",
    # )
    ax.legend()
    return ax


ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=False)

plt.show()
