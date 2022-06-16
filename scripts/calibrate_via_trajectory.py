#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from time import time
# import pcl

# projectile imports 
import random
from shield_planner_msgs.srv import TestbedProjectile
from shield_planner_msgs.msg import Projectile
import std_msgs.msg
import datetime
import math
import rospkg
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


g_received_sc = False
g_received_ki = False
msg_sc = None
msg_ki = None

def callback_sc(msg):
    global g_received_sc, msg_sc
    print("Received SC")
    g_received_sc = True
    msg_sc = msg

def callback_ki(msg):
    global g_received_ki, msg_ki
    print("Received KI")
    g_received_ki = True
    msg_ki = msg

def calibrate_trajectories():
    ki_vel = np.array([msg_ki.velocity.x, msg_ki.velocity.y, msg_ki.velocity.z])
    sc_vel = np.array([msg_sc.velocity.x, msg_sc.velocity.y, msg_sc.velocity.z])
    ki_vel_dir = ki_vel/np.linalg.norm(ki_vel)
    sc_vel_dir = sc_vel/np.linalg.norm(sc_vel)
    axis = np.cross(sc_vel_dir,ki_vel_dir)
    angle = np.dot(sc_vel_dir,ki_vel_dir)

    ki_pos = np.array([msg_ki.position.x, msg_ki.position.y, msg_ki.position.z])
    sc_pos = np.array([msg_sc.position.x, msg_sc.position.y, msg_sc.position.z])

    print("Angle", angle)
    print("Axis", axis)
    print("KI", ki_pos, ki_vel)
    print("SC", sc_pos, sc_vel)


    
def listener():
    rospy.init_node('calibrate_via_trajectory', anonymous=True)
    
    rospy.Subscriber("projectile", Projectile, callback_sc)
    rospy.Subscriber("projectile_2", Projectile, callback_ki)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        if g_received_ki and g_received_sc:
            calibrate_trajectories()
        rate.sleep()

if __name__ == '__main__':
    listener()