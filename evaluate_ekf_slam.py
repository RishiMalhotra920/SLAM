import numpy as np

from ekf_slam import EKFSLAM

ekf_slam_instance = EKFSLAM()
num_runs = 500
total_trajectory_rmse = 0
total_landmark_rmse = 0
for init in range(num_runs):
    true_robot_trajectory, dead_reckoning_robot_trajectory, belief_robot_trajectory, landmark_beliefs_over_time = (
        ekf_slam_instance.simulate_ekf_slam_known_correspondences()
    )

    trajectory_rmse, landmark_rmse = ekf_slam_instance.calculate_rmse(
        np.array(true_robot_trajectory), np.array(belief_robot_trajectory), ekf_slam_instance.true_landmarks, np.array(landmark_beliefs_over_time)
    )

    print("this is the trajectory rmse", trajectory_rmse)
    total_trajectory_rmse += trajectory_rmse
    total_landmark_rmse += landmark_rmse

print(f"Average Trajectory RMSE after {num_runs} runs: {total_trajectory_rmse / num_runs}")
print(f"Average Landmark RMSE after {num_runs} runs: {total_landmark_rmse / num_runs}")
