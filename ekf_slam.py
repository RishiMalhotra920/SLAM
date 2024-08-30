import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 10))

true_landmarks = np.array([[5, 5], [-4, 6], [-6, -4]])

true_robot_trajectory = []
dead_reckoning_robot_trajectory = []
belief_robot_trajectory = []

state_len = 3 + 2 * len(true_landmarks)
true_state_prev = np.zeros((state_len))
mu_prev = np.zeros(state_len)
sigma_prev = np.eye(state_len)
landmarks_seen: set[int] = set()
N_t_prev = 0
R_t = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
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
    delta_t = 1
    x, y, theta = state[:3]
    v, omega = u_t

    # Check for small omega values
    if abs(omega) < 1e-6:  # You can adjust this threshold as needed
        # Use a different motion model for straight-line motion
        motion_array = np.array([v * np.cos(theta) * delta_t, v * np.sin(theta) * delta_t, omega * delta_t])
    else:
        # Use the original motion model for curved motion
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
    mu_prev: np.ndarray, sigma_prev: np.ndarray, u_t: np.ndarray, z_t: np.ndarray, c_t: np.ndarray, R_t: np.ndarray, landmarks_seen: set
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # returns mu_bar_t, sigma_bar_t, mu_t, sigma_t

    # maintenance
    v_t, omega_t = u_t
    delta_t = 1

    # step 1
    F_x = np.zeros((3, state_len))
    F_x[:3, :3] = np.eye(3)

    # step 2
    if abs(omega_t) < 1e-6:
        motion_array = np.array([v_t * np.cos(mu_prev[2]) * delta_t, v_t * np.sin(mu_prev[2]) * delta_t, omega_t * delta_t])
    else:
        motion_array = np.array(
            [
                -v_t / omega_t * np.sin(mu_prev[2]) + v_t / omega_t * np.sin(mu_prev[2] + omega_t * delta_t),
                v_t / omega_t * np.cos(mu_prev[2]) - v_t / omega_t * np.cos(mu_prev[2] + omega_t * delta_t),
                omega_t * delta_t,
            ]
        )

    mu_bar_t = mu_prev + F_x.T @ motion_array

    # step 3 (Jacobian calculation)
    if abs(omega_t) < 1e-6:
        jacobian_delta = np.array([[0, 0, -v_t * np.sin(mu_prev[2]) * delta_t], [0, 0, v_t * np.cos(mu_prev[2]) * delta_t], [0, 0, 0]])
    else:
        jacobian_delta = np.array(
            [
                [0, 0, -v_t / omega_t * np.cos(mu_prev[2]) + v_t / omega_t * np.cos(mu_prev[2] + omega_t * delta_t)],
                [0, 0, -v_t / omega_t * np.sin(mu_prev[2]) + v_t / omega_t * np.sin(mu_prev[2] + omega_t * delta_t)],
                [0, 0, 0],
            ]
        )
    # step 4
    G_t = np.eye(state_len) + F_x.T @ np.array(jacobian_delta) @ F_x

    # step 5
    sigma_bar_t = G_t @ sigma_prev @ G_t.T + F_x.T @ R_t @ F_x

    mu_bar_t_to_return = mu_bar_t.copy()
    sigma_bar_t_to_return = sigma_bar_t.copy()

    # step 6
    Q_t = np.eye(2)
    sigma_r = 0.3
    sigma_theta = 0.3
    np.fill_diagonal(Q_t, [sigma_r**2, sigma_theta**2])
    # feature idx is the feature that we have observed. in this case, we fully observe all landmarks at every step

    # step 7
    for feature_idx in range(len(true_landmarks)):
        # the true landmark index that we are observing. in this case, it's the same as the feature index
        # step 8
        j = c_t[feature_idx]
        z_t_i = z_t[feature_idx]  # shape (2,)

        # step 9
        if j not in landmarks_seen:
            r_t_i = z_t_i[0]
            phi_t_i = z_t_i[1]

            # step 10
            mu_bar_t[3 + 2 * j : 3 + 2 * j + 2] += np.array([r_t_i * np.cos(phi_t_i + mu_bar_t[2]), r_t_i * np.sin(phi_t_i + mu_bar_t[2])])

            landmarks_seen.add(j)

        # step 11 endif
        # step 12 - based on just the motion, calculate expected measurement in x, y
        delta = mu_bar_t[3 + 2 * j : 3 + 2 * j + 2] - mu_bar_t[:2]
        # step 13
        q = np.dot(delta, delta)
        # step 14 - convert expected measurement in r, theta
        z_t_i_hat = np.array([np.sqrt(q), normalize_angle(np.arctan2(delta[1], delta[0]) - mu_bar_t[2])])

        # step 15
        F_x_j = np.zeros((5, state_len))
        F_x_j[:3, :3] = np.eye(3)
        F_x_j[3:5, 3 + 2 * j : 3 + 2 * j + 2] = np.eye(2)

        # step 16
        H_t_i = (
            1
            / q
            * (
                np.array(
                    [
                        [-np.sqrt(q) * delta[0], -np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                        [delta[1], -delta[0], -q, -delta[1], delta[0]],
                    ]
                )
            )
            @ F_x_j
        )

        # step 17
        K_t_i = sigma_bar_t @ H_t_i.T @ np.linalg.inv(H_t_i @ sigma_bar_t @ H_t_i.T + Q_t)
        mu_bar_t += K_t_i @ (z_t_i - z_t_i_hat)
        sigma_bar_t = (np.eye(state_len) - K_t_i @ H_t_i) @ sigma_bar_t

    mu_t = mu_bar_t
    sigma_t = sigma_bar_t

    return mu_bar_t_to_return, sigma_bar_t_to_return, mu_t, sigma_t


def normalize_angle(theta):
    # keeps angles between -pi and pi
    # kinda wild that this works
    return np.arctan2(np.sin(theta), np.cos(theta))


def animate(frame):
    global mu_prev, sigma_prev, true_state_prev, N_t_prev, true_landmarks, R_t, landmarks_seen
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # make the state transition

    u_t_gt, u_t_sensor = get_u_t(frame)
    true_state = get_true_state(true_state_prev, u_t_gt)
    z_t, c_t = get_z_t(true_state_prev, true_landmarks)

    mu_bar, sigma_bar, mu_t, sigma_t = run_ekf_slam_known_correspondences(mu_prev, sigma_prev, u_t_sensor, z_t, c_t, R_t, landmarks_seen)

    true_robot_trajectory.append(true_state[:3])
    dead_reckoning_robot_trajectory.append(mu_bar[:3])
    belief_robot_trajectory.append(mu_t[:3])

    true_state_prev = true_state
    mu_prev = mu_t  # mu_bar earlier
    sigma_prev = sigma_t  # sigma_bar earlier

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

    # Draw the robot belief trajectory
    ax.plot(
        [x[0] for x in belief_robot_trajectory],
        [x[1] for x in belief_robot_trajectory],
        c="green",
        label="Robot Belief Trajectory",
    )

    # Draw the landmark beliefs
    landmark_beliefs = mu_t[3:].reshape(-1, 2)
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
