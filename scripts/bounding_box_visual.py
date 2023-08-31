#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
import roslib
roslib.load_manifest('laser_assembler')
from laser_assembler.srv import *

# Method marcro: 0 = bounding box, 1 = color filtering
METHOD_ID=1

def callback(pc):
    # pc_save = args[1]
    # if pc.header.stamp.secs + 1 > rospy.get_time():
    #     pc_save = PointCloud2()
    #     pc_save.header.frame_id = "odom_combined"
    # pc_save.header.stamp = rospy.Time.now()
    # pc_save.data.
    try:
        assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)
        resp = assemble_scans(rospy.Time(2,0), rospy.get_rostime())
        # print("Got cloud with %u points" % len(resp.cloud.points))
        pc_pub = rospy.Publisher("/pc_vis", PointCloud2, queue_size=1)
        pc_pub.publish(resp.cloud)
    except rospy.ServiceException as e:
        print("Servie call failed: %s"%e)

    

def listener():
    # Publish the pc batch in track_ball instead
    rospy.init_node('filtered_pc_listener')
    # rospy.wait_for_service('assemble_scans2')
    # print("Got Service!")
    # # rospy.Subscriber("/zed2i/filtered_point_cloud", PointCloud2, callback)
    # while not rospy.is_shutdown():
    #     try:
    #         assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)
    #         resp = assemble_scans(rospy.Time(2,0), rospy.get_rostime())
    #         # print("Got cloud with %u points" % len(resp.cloud.points))
    #         pc_pub = rospy.Publisher("/pc_vis", PointCloud2, queue_size=1)
    #         pc_pub.publish(resp.cloud)
    #     except rospy.ServiceException as e:
    #         print("Servie call failed: %s"%e)
        # print(rospy.Time.now())
        # rospy.rostime.wallsleep(2.0)


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
    elif METHOD_ID == 1:
        listener()
