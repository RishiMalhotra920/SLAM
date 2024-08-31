import numpy as np


class EKFSLAM:
    def __init__(self):
        # true_landmarks = np.array([[5, 5], [-4, 6], [-6, -4], [7, -3]])
        self.true_landmarks = np.random.uniform(-10, 10, (4, 2))

        self.state_len = 3 + 2 * len(self.true_landmarks)

        # set the stds for u - actual motion noise
        self.u_v_std = 0.05
        self.u_omega_std = 0.05

        # set the stds for R - expected motion noise
        R_t_x_std = 0.1
        R_t_y_std = 0.1
        R_t_theta_std = 0.1
        self.R_t = np.array([[R_t_x_std**2, 0.0, 0.0], [0.0, R_t_y_std**2, 0.0], [0.0, 0.0, R_t_theta_std**2]])

        # set the stds for z_t - actual observation noise
        self.z_t_r_std = 0.05
        self.z_t_theta_std = 0.05

        # set the stds for Q - expected observation noise
        self.Q_std_r = 0.05
        self.Q_std_theta = 0.05

    def get_u_t(
        self,
        t,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        u_v and u_omega are control inputs which come with some noise
        v, omega are the true control inputs

        returns the true control input and the noisy control input
        """
        u_v = 0.25
        u_omega = 0.1
        v_gt = np.random.normal(u_v, self.u_v_std)
        omega_gt = np.random.normal(u_omega, self.u_omega_std)

        return np.array([v_gt, omega_gt]), np.array([u_v, u_omega])

    def get_true_state(self, state: np.ndarray, u_t: np.ndarray):
        delta_t = 1
        x, y, theta = state[:3]
        v, omega = u_t

        # Check for small omega values
        if abs(omega) < 1e-6:  # You can adjust this threshold as needed
            # Use a different motion model for straight-line motion
            print("straight line motion")
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

    def get_z_t(
        self,
        true_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """given true state, return the sensor measurements"""

        r_t = np.linalg.norm(self.true_landmarks - true_state[:2], axis=1)
        angle_to_landmark = np.arctan2(self.true_landmarks[:, 1] - true_state[1], self.true_landmarks[:, 0] - true_state[0])
        theta_t = self.normalize_angle(angle_to_landmark - true_state[2])
        z_t = np.array([r_t, theta_t]).T  # shape (n_landmarks, 2)

        sensor_noise = np.zeros_like(z_t)
        sensor_noise[:, 0] = np.random.normal(0, self.z_t_r_std, z_t.shape[0])
        sensor_noise[:, 1] = np.random.normal(0, self.z_t_theta_std, z_t.shape[0])

        z_t_sensor = z_t + sensor_noise
        # np.random.normal(0, 0.1, z_t.shape)

        c_t = np.arange(len(self.true_landmarks))

        # return z_t, c_t

        return z_t_sensor, c_t

    def get_dead_reckoning_state(
        self,
        mu_prev: np.ndarray,
        sigma_prev: np.ndarray,
        u_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mu_prev = mu_prev.copy()
        u_t = u_t.copy()

        v_t, omega_t = u_t
        delta_t = 1

        # step 1
        F_x = np.zeros((3, self.state_len))
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
        G_t = np.eye(self.state_len) + F_x.T @ np.array(jacobian_delta) @ F_x

        # step 5
        sigma_bar_t = G_t @ sigma_prev @ G_t.T + F_x.T @ self.R_t @ F_x

        return mu_bar_t, sigma_bar_t

    def run_ekf_slam_known_correspondences(
        self,
        mu_prev: np.ndarray,
        sigma_prev: np.ndarray,
        u_t: np.ndarray,
        z_t: np.ndarray,
        c_t: np.ndarray,
        landmarks_seen: set,
    ) -> tuple[np.ndarray, np.ndarray]:
        # steps 1-5
        mu_bar_t, sigma_bar_t = self.get_dead_reckoning_state(mu_prev, sigma_prev, u_t)

        # step 6
        Q_t = np.eye(2)

        np.fill_diagonal(Q_t, [self.Q_std_r**2, self.Q_std_theta**2])
        # feature idx is the feature that we have observed. in this case, we fully observe all landmarks at every step

        # step 7
        for feature_idx in range(len(self.true_landmarks)):
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
            z_t_i_hat = np.array([np.sqrt(q), self.normalize_angle(np.arctan2(delta[1], delta[0]) - mu_bar_t[2])])

            # step 15
            F_x_j = np.zeros((5, self.state_len))
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
            sigma_bar_t = (np.eye(self.state_len) - K_t_i @ H_t_i) @ sigma_bar_t

        mu_t = mu_bar_t
        sigma_t = sigma_bar_t

        return mu_t, sigma_t

    def normalize_angle(self, theta):
        # keeps angles between -pi and pi
        # kinda wild that this works
        return np.arctan2(np.sin(theta), np.cos(theta))

    def calculate_rmse(self, true_trajectory: np.ndarray, belief_trajectory: np.ndarray, true_landmarks: np.ndarray, belief_landmarks: np.ndarray):
        trajectory_rmse = np.sqrt(np.mean((true_trajectory - belief_trajectory) ** 2))
        landmark_rmse = np.sqrt(np.mean((true_landmarks - belief_landmarks) ** 2))
        return trajectory_rmse, landmark_rmse

    def simulate_ekf_slam_known_correspondences_step(
        self,
        frame,
        mu_prev,
        sigma_prev,
        mu_dead_reckoning_prev,
        sigma_dead_reckoning_prev,
        true_state_prev,
        landmarks_seen,
    ):
        # make the state transition

        u_t_gt, u_t_sensor = self.get_u_t(frame)
        true_state = self.get_true_state(true_state_prev, u_t_gt)
        z_t, c_t = self.get_z_t(true_state_prev)

        mu_dead_reckoning, sigma_dead_reckoning = self.get_dead_reckoning_state(mu_dead_reckoning_prev, sigma_dead_reckoning_prev, u_t_sensor)
        mu_t, sigma_t = self.run_ekf_slam_known_correspondences(mu_prev, sigma_prev, u_t_sensor, z_t, c_t, landmarks_seen)

        return true_state, mu_dead_reckoning, sigma_dead_reckoning, mu_t, sigma_t

    def simulate_ekf_slam_known_correspondences(self) -> tuple[list, list, list, list]:
        landmarks_seen = set()  # type: ignore
        true_state_prev = np.zeros((self.state_len))
        mu_prev = np.zeros(self.state_len)
        # sigma_prev = np.eye(state_len)
        sigma_prev = np.zeros((self.state_len, self.state_len))
        sigma_prev[:3, :3] = np.eye(3)
        np.fill_diagonal(sigma_prev[3:, 3:], 10)

        mu_dead_reckoning_prev = np.zeros(self.state_len)
        # sigma_dead_reckoning_prev = np.eye(state_len)
        sigma_dead_reckoning_prev = np.zeros((self.state_len, self.state_len))
        sigma_dead_reckoning_prev[:3, :3] = np.eye(3)
        np.fill_diagonal(sigma_dead_reckoning_prev[3:, 3:], 1000)

        true_robot_trajectory = []
        dead_reckoning_robot_trajectory = []
        belief_robot_trajectory = []
        landmark_beliefs_over_time = []

        for t in range(200):
            true_state, mu_dead_reckoning, sigma_dead_reckoning, mu_t, sigma_t = self.simulate_ekf_slam_known_correspondences_step(
                t, mu_prev, sigma_prev, mu_dead_reckoning_prev, sigma_dead_reckoning_prev, true_state_prev, landmarks_seen
            )

            true_robot_trajectory.append(true_state[:3])
            dead_reckoning_robot_trajectory.append(mu_dead_reckoning[:3])
            belief_robot_trajectory.append(mu_t[:3])
            landmark_beliefs_over_time.append(mu_t[3:].reshape(-1, 2))

            mu_prev = mu_t
            sigma_prev = sigma_t
            mu_dead_reckoning_prev = mu_dead_reckoning
            sigma_dead_reckoning_prev = sigma_dead_reckoning
            true_state_prev = true_state

        return true_robot_trajectory, dead_reckoning_robot_trajectory, belief_robot_trajectory, landmark_beliefs_over_time
