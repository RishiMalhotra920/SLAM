import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import EKFSLAM

ekf_slam_instance = EKFSLAM()

fig, ax = plt.subplots(figsize=(10, 10))
plt.ion()  # Turn on interactive mode
fig.show()

num_frames = 200
true_robot_trajectory, dead_reckoning_robot_trajectory, belief_robot_trajectory, landmark_beliefs_over_time = (
    ekf_slam_instance.simulate_ekf_slam_known_correspondences()
)
print("this is landmarks", ekf_slam_instance.true_landmarks)

for frame in range(num_frames):
    true_state = true_robot_trajectory[frame]
    mu_t = belief_robot_trajectory[frame]
    landmark_beliefs = landmark_beliefs_over_time[frame]

    print(true_state - mu_t)

    if np.sum(true_state[:2] - mu_t[:2]) > 2:
        print("this is the frame", frame)
        print("this is the true state", true_state)
        print("this is the belief state", mu_t)
        print("this is the landmark beliefs", landmark_beliefs)
        input()

    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Draw the true landmarks
    ax.scatter(
        ekf_slam_instance.true_landmarks[:, 0],
        ekf_slam_instance.true_landmarks[:, 1],
        c="blue",
        label="Landmark Ground Truth",
    )

    # Draw the robot
    arrow_length = 0.05
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
        [x[0] for x in true_robot_trajectory[: frame + 1]],
        [x[1] for x in true_robot_trajectory[: frame + 1]],
        c="blue",
        label="True Robot Trajectory",
    )

    # Draw the robot dead reckoning trajectory
    ax.plot(
        [x[0] for x in dead_reckoning_robot_trajectory[: frame + 1]],
        [x[1] for x in dead_reckoning_robot_trajectory[: frame + 1]],
        c="black",
        label="Dead reckoning Robot Trajectory",
    )

    # Draw the robot belief trajectory
    ax.plot(
        [x[0] for x in belief_robot_trajectory[: frame + 1]],
        [x[1] for x in belief_robot_trajectory[: frame + 1]],
        c="green",
        label="Robot Belief Trajectory",
    )

    # Draw the landmark beliefs

    ax.scatter(
        landmark_beliefs[: frame + 1][:, 0],
        landmark_beliefs[: frame + 1][:, 1],
        c="gray",
        label="Landmark Beliefs",
    )
    ax.legend()

    # Update the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Optional: add a small delay to control the animation speed
    plt.pause(0.003)

# Keep the plot open after the animation is finished
plt.ioff()  # Turn off interactive mode
plt.show()
