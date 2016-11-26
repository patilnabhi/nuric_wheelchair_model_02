## Smart Wheelchair - Estimator for Caster Wheel Orientation (Project in development)

* **Final Project: Developing shared autonomy behaviors for a smart wheelchair at Rehabilitation Institute of Chicago (RIC) in collaboration with Prof. Brenna Argall** 
* Currently, developed a 3D URDF model of wheelchair in simulation, a wall-following behavior in Python/C++, and an estimator for Caster Wheels Orientation (CWOs) and studied the existing code base
* Developed a kinematic/dynamic model for the wheelchair, taking into account CWOs and friction between wheels and ground
* Implementing an Unscented Kalman Filter (UKF) algorithm in determining & correcting the state of the wheelchair, given the CWOs and kinematic/dynamic model of the wheelchair