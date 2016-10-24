#!/usr/bin/python

import sys
import roslib
from nuric_system.msg import LowLevelCommand
from nuric_system.srv import LowLevelCommands
import rospy
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan


class WallFollowerNode():
    "Wall follower node"

    def __init__(self):
        "WallFollowerNode constructor"
        rospy.init_node('wall_follower')
        
        self.linear_rate = 0.4
        self.angular_rate = 0.35
        self.Rtemp = 150
        self.Rtemp2 = 450
        self.R1 = 310
        self.R2 = 315
        self.R3 = 320
        self.R4 = 325
        self.linear_wall_follower_rate = 0.8
        self.angular_wall_follower_rate = 0.06

        self.send_cmd = LowLevelCommand()
        self.cmd_client = rospy.ServiceProxy('send_command', LowLevelCommands)
        self.joy = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.llc = rospy.Publisher("/user_cmd", LowLevelCommand, queue_size=1)
        self.laser_scan = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.pub()
            rate.sleep()

    def pub(self):
        self.send_cmd.header.stamp = rospy.Time.now()
        self.send_cmd.header.frame_id = "base_footprint"
        self.send_cmd.goal_source = "teleoperation"
        self.llc.publish(self.send_cmd)
        self.cmd_client(self.send_cmd)

    def joy_callback(self, joy):

        if len(joy.buttons) == 0:
            return
        # initialize commands...
        latest_cmd = LowLevelCommand()
        latest_cmd.command.linear.x = self.linear_rate * joy.axes[1]
        latest_cmd.command.angular.z = self.angular_rate * joy.axes[0]
        
        self.send_cmd = latest_cmd

    def laser_scan_callback(self, laser_scan):
        if laser_scan.ranges[self.Rtemp] < 3.0:
            R0 = self.Rtemp;
        else:
            R0 = self.Rtemp2;
        
        rospy.loginfo("Range at %d: %f", R0, laser_scan.ranges[R0])
        rospy.loginfo("Range at %d: %f", self.R1, laser_scan.ranges[self.R1])
        rospy.loginfo("Range at %d: %f", self.R2, laser_scan.ranges[self.R2])
        rospy.loginfo("Range at %d: %f", self.R3, laser_scan.ranges[self.R3])
        rospy.loginfo("Range at %d: %f", self.R4, laser_scan.ranges[self.R4])

        latest_cmd = LowLevelCommand()

        if (not self.isValid(laser_scan.ranges[R0])) and self.isValid(laser_scan.ranges[self.R1]) and self.isValid(laser_scan.ranges[self.R2]) and self.isValid(laser_scan.ranges[self.R3]) and self.isValid(laser_scan.ranges[self.R4]):
            rospy.loginfo("Following wall...")
            

            if laser_scan.ranges[R0] > 1.4 and laser_scan.ranges[R0] < 1.5:
                rospy.loginfo("Moving straight...")
                self.send_cmd.command.linear.x = self.linear_wall_follower_rate
            
            elif laser_scan.ranges[R0] > 1.48 and laser_scan.ranges[R0] < 3.0:
                rospy.loginfo("Adjusting Orient...");
                if R0 == self.Rtemp: 
                    self.send_cmd.command.angular.z = -self.angular_wall_follower_rate
                if R0 == self.Rtemp2: 
                    self.send_cmd.command.angular.z = self.angular_wall_follower_rate

            elif laser_scan.ranges[R0] < 1.42:
                rospy.loginfo("Adjusting Orient...")
                if R0 == self.Rtemp:
                    self.send_cmd.command.angular.z = self.angular_wall_follower_rate
                if R0 == self.Rtemp2:
                    self.send_cmd.command.angular.z = -self.angular_wall_follower_rate
        
        else:
            rospy.loginfo("Please go near a wall!")


    def isValid(self, range_val):
        if range_val < 3.0:
            return False
        else:
            return True

def main():
    WallFollowerNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException: pass

if __name__ == '__main__':
    sys.exit(main())