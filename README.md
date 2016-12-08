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

* The UKF algorithm implementation consists of 3 main steps, as outlined below –

	(a) Initialize:
		* Initialize state and controls for the wheelchair (mean and covariance)

	(b) Predict:
		* Generate sigma points using Julier’s Scaled Sigma Point algorithm
		* Pass each sigma points through the dynamic motion model to from a new prior
		* Determine mean and covariance of new prior through unscented transform

	(c) Update:
		* Get odometry data (measurement of pose of wheelchair)
		* Convert the sigma points of prior into expected measurements (points corresponding to pose of wheelchair – x, y  and \theta  are chosen)
		* Compute mean and covariance of converted sigma points through unscented transform
		* Compute residual and Kalman gain
		* Determine new estimate for the state with new covariance


* The UKF code (Python) is produced below (Click on functions to look at its complete implementation): 

```
def fx(x, dt):	
	sol = ode2(x)
	return np.array(sol)

def hx(x):
	return np.array([x[3], x[2], normalize_angle(x[4])])

```

```
points = JulierSigmaPoints(n=7, kappa=-4., sqrt_method=None)

kf = UKF(dim_x=7, dim_z=3, dt, fx, hx, points, 
			sqrt_fn=None, x_mean_fn=state_mean, z_mean_fn=meas_mean, 
			residual_x, residual_z)

x0 = np.array(self.ini_val)

kf.x = x0
kf.Q *= np.diag([.0001, .0001, .0001, .0001, .0001, .01, .01])
kf.P *= 0.000001
kf.R *= 0.0001

move_time = 4.0
start = rospy.get_time()

while (rospy.get_time() - start < move_time) and not rospy.is_shutdown():	
	pub_twist.publish(wheel_cmd)

	z = np.array([odom_x, odom_y, odom_theta])
	zs.append(z)

	kf.predict()
	kf.update(z)

	xs.append(kf.x)

```




