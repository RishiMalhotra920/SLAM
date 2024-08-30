# SLAM


https://github.com/user-attachments/assets/10195c25-108e-4009-bf12-0c59bd279666


The robot navigates a predefined path on a 2d grid, determining the map and its location on the map. We assume known correspondences between the observations the robot makes and the true landmarks.

The robot receives control signals that tell it to move in circles (its dead reckoning trajectory). However, these signals are noisy and the robot ends up going on an adventure (the true trajectory). I use SLAM to determine the robot's belief trajectory using noisy control inputs and noisy observations that track its true trajectory.

I artificially add gaussian noise to the robot's control inputs and the robot's observations. I then go ahead and model this noise with my EKF SLAM process.

Specifically, I implement the `EKF_SLAM_known_correspondences` algorithm on page 314 in the Probabilistic Robotics textbook (cited below).

The code is not the cleanest but it's for learning purposes so that's okay :)

Some potential improvements:

- I model the robot's control input and observation by adding a fixed amount of noise to its control and observation inputs. Noise is better represented as a proportion of the actual sensor data. So instead of adding gaussian noise with a fixed std 0.01, I would add gaussian noise with std 5% of the sensor value.
- There may be minor technical errors in the implementation. I can fix those.

```
@book{10.5555/1121596,
author = {Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter},
title = {Probabilistic Robotics (Intelligent Robotics and Autonomous Agents)},
year = {2005},
isbn = {0262201623},
publisher = {The MIT Press}
}
```
