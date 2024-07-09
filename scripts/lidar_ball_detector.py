#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np

pub1 = rospy.Publisher('/scan1_diff', Float64, queue_size = 10)
pub2 = rospy.Publisher('/scan2_diff', Float64, queue_size = 10)

previous_scan_1 = LaserScan()
previous_scan_2 = LaserScan()

def LidarCallback1(msg):
    global previous_scan_1
    if len(previous_scan_1.ranges) == 0:
        previous_scan_1 = msg
        return

    # ceiling filter
    ranges = np.array(previous_scan_1.ranges)
    filt_ranges = ranges[ranges < 2]

    # scan1_diff = np.linalg.norm(np.array(msg.ranges) - np.array(previous_scan_1.ranges))
    scan1_diff = np.linalg.norm(filt_ranges)
    previous_scan_1 = msg
    pub1.publish(scan1_diff)

def LidarCallback2(msg):
    global previous_scan_2
    if len(previous_scan_2.ranges) == 0:
        previous_scan_2 = msg
        return

    # ceiling filter
    ranges = np.array(previous_scan_2.ranges)
    filt_ranges = ranges[ranges < 2]

    # scan2_diff = np.linalg.norm(np.array(msg.ranges) - np.array(previous_scan_2.ranges))
    scan2_diff = np.linalg.norm(filt_ranges)
    previous_scan_2 = msg
    pub2.publish(scan2_diff)

def LidarListener():
    rospy.init_node('LidarListener', anonymous=True)
    sub1 = rospy.Subscriber('/inner_scan', LaserScan, LidarCallback1)
    sub2 = rospy.Subscriber('/outer_scan', LaserScan, LidarCallback2)
    rospy.spin()

if __name__ == '__main__':
    LidarListener()
