#include <ros/ros.h>
#include <nuric_system/NodeRegistration.h>
#include <nuric_system/LowLevelCommands.h>
#include <nuric_system/LowLevelCommand.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/LaserScan.h>

#include <iostream>
#include <fstream>

//#0: openni, 1:phidgets, 2:imu, 3:joy_node, 4:keyboard, 5:gui_command_widget, 6:odometry_node, 7:base_tf_node, 8:base_motor_comm, 9:doorway_assistance, 10:docking_assistance, 11:obstacle_avoidance, 12:teleoperation
#define NODE_NAME       "wall_follower"
#define NODE_TYPE       200

using namespace std;

ros::NodeHandle * nh;

nuric_system::LowLevelCommands latest_cmd;
nuric_system::LowLevelCommands joy_cmd;


int linear_, angular_;
double l_scale_, a_scale_;

ros::Publisher usr_cmd_pub;  // for system monitoring of user commanded velocities

void joy_cb(const sensor_msgs::Joy::ConstPtr& joy)
{
  latest_cmd.request.command.command.linear.x = l_scale_*joy->axes[linear_];
  latest_cmd.request.command.command.angular.z = a_scale_*joy->axes[angular_];
}

int R0;
int R1 = 310, R2 = 315, R3 = 320, R4 = 325;

bool isValid(float range_val) {
  if (range_val < 3.0) {
    return false;
  } else {
    return true;
  }
}

void scan_cb(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {
  if (scan_msg->ranges[150] < 3.0) { 
    R0 = 150; 
  } else {
    R0 = 450;
  }

  ROS_INFO("Range @ %d: %f", R0, scan_msg->ranges[R0]);
  ROS_INFO("Range @ %d: %f", R1, scan_msg->ranges[R1]);
  ROS_INFO("Range @ %d: %f", R2, scan_msg->ranges[R2]);
  ROS_INFO("Range @ %d: %f", R3, scan_msg->ranges[R3]);
  ROS_INFO("Range @ %d: %f", R4, scan_msg->ranges[R4]); 
    
  if (!isValid(scan_msg->ranges[R0]) && isValid(scan_msg->ranges[R1]) && isValid(scan_msg->ranges[R2]) && isValid(scan_msg->ranges[R3]) && isValid(scan_msg->ranges[R4])) {
    ROS_INFO("Following wall...");
    if (scan_msg->ranges[R0] > 1.4 && scan_msg->ranges[R0] < 1.5) {
      ROS_INFO("Moving straight...");
      latest_cmd.request.command.command.linear.x = 0.8;
    }
    if (scan_msg->ranges[R0] > 1.48 && scan_msg->ranges[R0] < 3.0) {
      ROS_INFO("Adjusting Orient...");
      if (R0 == 150) { latest_cmd.request.command.command.angular.z = -0.06; }
      if (R0 == 450) { latest_cmd.request.command.command.angular.z = 0.06; }

    }
    if (scan_msg->ranges[R0] < 1.42) {
      ROS_INFO("Adjusting Orient...");
      if (R0 == 150) { latest_cmd.request.command.command.angular.z = 0.06; }
      if (R0 == 450) { latest_cmd.request.command.command.angular.z = -0.06; }
    }
  } else {
    ROS_INFO("Please go near a wall!");
  }
}


int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, NODE_NAME);
  ros::NodeHandle n;
  nh = &n;

  linear_ = 1;
  angular_ = 0;

  a_scale_ = 0.35;
  l_scale_ = 0.4;

  ros::Subscriber joy_sub = nh->subscribe("joy", 1, joy_cb);
  usr_cmd_pub = nh->advertise<nuric_system::LowLevelCommand>("/user_cmd", 1);
  ros::ServiceClient command_client = n.serviceClient<nuric_system::LowLevelCommands>("send_command");

  ros::Subscriber sub = nh->subscribe("/scan", 1000, scan_cb);

  latest_cmd.request.command.header.frame_id = "base_footprint";
  latest_cmd.request.command.goal_source = "teleoperation";

  ros::Rate loop_rate(30.0); 

  //Control Loop and code
  while(ros::ok())
  {
    {
      latest_cmd.request.command.header.stamp = ros::Time::now();
      command_client.call(latest_cmd);
      usr_cmd_pub.publish(latest_cmd.request.command);
    }      
    ros::spinOnce(); 
    loop_rate.sleep();      
  }
}
