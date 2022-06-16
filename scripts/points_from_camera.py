#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from time import time
# import pcl
from shield_planner_msgs.srv import TestbedProjectile
from shield_planner_msgs.msg import Projectile


G_DEBUG = True
publisher = None
calibrate_frame = 5
x_pos_lim = 1.5
x_neg_lim = 0.0
z_pos_lim = 3.
z_neg_lim = 0.01
y_pos_lim = 0.75
y_neg_lim = -0.75



# def ros_to_pcl(ros_cloud):
#     """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

#         Args:
#             ros_cloud (PointCloud2): ROS PointCloud2 message

#         Returns:
#             pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
#     """
#     points_list = []

#     for data in pc2.read_points(ros_cloud, skip_nans=True):
#         points_list.append([data[0], data[1], data[2], data[3]])

#     pcl_data = pcl.PointCloud_PointXYZRGB()
#     pcl_data.from_list(points_list)

#     return pcl_data 

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tostring()

    return msg

def filter_ball_points(ros_cloud):
    t1 = rospy.Time.now().to_sec()
    global y_pos_lim, y_neg_lim, z_pos_lim, z_neg_lim, calibrate_frame
    if calibrate_frame>0:
        print("Calibrating")
        for data in pc2.read_points(ros_cloud, skip_nans=True):
            x, y, z = data[0], data[1] , data[2]
            if (z<z_pos_lim and z>0.05 and x>x_neg_lim and x<x_pos_lim):
                if y>0 and y<y_pos_lim:
                    y_pos_lim=y
                elif y<0 and y>y_neg_lim:
                    y_neg_lim=y
        calibrate_frame-=1
        if calibrate_frame==0:
            y_pos_lim = y_pos_lim*0.95
            y_neg_lim = y_neg_lim*0.95
            print("Calibration complete: Bounds", x_pos_lim, x_neg_lim, y_pos_lim, y_neg_lim, z_pos_lim, z_neg_lim)
        return

    point_list = []
    pc = rnp.numpify(ros_cloud)
    points = np.zeros((pc.shape[0],3))
    points[:,0] = pc['x']
    points[:,1] = pc['y']
    points[:,2] = pc['z']

    mask = np.concatenate([points[:,0:1]>x_neg_lim, points[:,0:1]<x_pos_lim, 
                           points[:,1:2]>y_neg_lim, points[:,1:2]<y_pos_lim, 
                           points[:,2:3]>z_neg_lim, points[:,2:3]<z_pos_lim], axis=1)
    mask = np.all(mask, axis=1)
    
    ball_points = points[mask,:]
    t2 = rospy.Time.now().to_sec()
    
    if(ball_points.shape[0]==0):
        return

    ball_position = np.mean(ball_points, axis=0)
    print(ball_position)


    if G_DEBUG:
        dtype_list = rnp.point_cloud2.fields_to_dtype(ros_cloud.fields, ros_cloud.point_step)
        filtered_msg = xyzrgb_array_to_pointcloud2(ball_points,
                                                   np.zeros(ball_points.shape, dtype=np.float32),
                                                   ros_cloud.header.stamp,
                                                   ros_cloud.header.frame_id,
                                                   ros_cloud.header.seq)
        publisher.publish(filtered_msg)
        t3 = rospy.Time.now().to_sec()
        print(t3-t1, "timediff")



def callback(msg):
    filter_ball_points(msg)

    
def listener():
    global publisher

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/sc/depth/points", PointCloud2, callback)
    publisher = rospy.Publisher('/sc/rgbd/filtered_points', PointCloud2, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()