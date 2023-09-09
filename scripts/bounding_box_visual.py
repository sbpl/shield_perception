#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import roslib
roslib.load_manifest('laser_assembler')
from laser_assembler.srv import *

# Method marcro: 0 = bounding box, 1 = color filtering
METHOD_ID=0


def visiualizedBox():
    rospy.init_node('shield_perception_bound')

    x_min = rospy.get_param("/zed_filtered_x/filter_limit_min")
    x_max = rospy.get_param("/zed_filtered_x/filter_limit_max")
    y_min = rospy.get_param("/zed_filtered_y/filter_limit_min")
    y_max = rospy.get_param("/zed_filtered_y/filter_limit_max")
    z_min = rospy.get_param("/zed_filtered/filter_limit_min")
    z_max = rospy.get_param("/zed_filtered/filter_limit_max")

    marker_pub = rospy.Publisher("/boundingbox_marker", Marker, queue_size = 2)

    marker = Marker()

    marker.header.frame_id = "odom_combined"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 1
    marker.id = 0

    # Set the scale of the marker
    marker.scale.x = x_max-x_min
    marker.scale.y = y_max-y_min
    marker.scale.z = z_max-z_min

    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.4

    # Set the pose of the marker
    marker.pose.position.x = (x_max+x_min)/2
    marker.pose.position.y = (y_max+y_min)/2
    marker.pose.position.z = (z_max+z_min)/2
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    while not rospy.is_shutdown():
        marker_pub.publish(marker)
        rospy.rostime.wallsleep(1.0)

if __name__ == '__main__':
    if METHOD_ID == 0:
        visiualizedBox()
