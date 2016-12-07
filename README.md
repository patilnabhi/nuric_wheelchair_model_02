## Smart Wheelchair - Estimator for Caster Wheel Orientation (Project in development)

#### Goal:
* Implement an Unscented Kalman FIlter (UKF) algorithm for accurate estimation of Caster Wheel Orientations (CWOs) and pose of a robotic wheelchair 
* Mentor: Prof. Brenna Argall
* Project is based in the assistive & rehabilitation robotics laboratory (argallab) located within the Rehabilitation Institute of Chicago (RIC)

#### Project Objectives:

* To study existing code structure and implement a wall-following behavior
* To design & simulate a 3D model of new wheelchair in ROS Gazebo and Rviz
* To research and implement a model in order to estimate wheelchair’s CWOs
* To implement an UKF algorithm for accurate estimation of CWOs and the pose of the wheelchair

#### Documentation Overview:

* This documentation explains the code structure and address the following 2 topics -

#####A. 3D model of new wheelchair

* The relevant files are present in 2 main directories, namely  `urdf` and `meshes`

* `urdf` : This directory contains the `xacro` files required to build the 3D model in simulation.
* Main highlights -
	* `joint_states` are published using the `gazebo_ros_control` plugin (particularly, `libgazebo_ros_joint_state_publisher.so` plugin)
	* The differential drive controller uses the `libgazebo_ros_diff_drive.so` plugin
	* The hokuyo laser controller uses `libgazebo_ros_laser.so` plugin to gather laser-scan data
	* The kinect camera controller uses `libgazebo_ros_openni_kinect.so` plugin to generate `rgb` and `depth` data

* `meshes` directory contain the collada `.dae` files of the wheelchair

* Raw SolidWorks files `.SLDPRT & .SLDASM` are available in the `3d_model_sw` directory
	* Blender software is used to convert the `.SLDPRT & .SLDASM` files into collada `.dae` files for URDF compatibility


#####B. UKF implementation for estimation of CWOs

```

def func(a, b):
	
	return b

```




http://stackoverflow.com/questions/11256433/how-to-show-math-equations-in-general-githubs-markdownnot-githubs-blog


