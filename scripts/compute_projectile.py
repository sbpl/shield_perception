#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from time import time

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
import tf


G_DEBUG = True
publisher = None
projectile_marker_pub = None
projectile_msg_pub = None
projectile_marker_time_pub = None
projectile_time_msg_pub = None
calibrate_frame = 5
# x_pos_lim = 1.5
# x_neg_lim = 0.0
# z_pos_lim = 3.
# z_neg_lim = 0.01
# y_pos_lim = 0.75
# y_neg_lim = -0.75

x_cen = -0.4515569
x_width = 0.768414*0.8
y_cen = -0.0709
y_width = 1.2218*0.97

x_pos_lim = x_cen + x_width/2.0
x_neg_lim = x_cen - x_width/2.0
z_pos_lim = 3.
z_neg_lim = 0.01
y_pos_lim = y_cen + y_width/2.0
y_neg_lim = y_cen - y_width/2.0

A_FP_SC = None
R_SC_KI = None
T_SC_KI = None
tf_trans = None
tf_rot = None
P_SC_B = np.ones((4,1))

vel_scale = 1.0
z_offset = 0.0
x_offset = 0.0
y_offset = 0.0
roll_offset = 0.0
pitch_offset = 0.0
yaw_offset = 0.0
gravity = -9.80665
drag_coeff = 0.015
mass_ball = 0.03135
res_yaw_A = np.eye(4)
res_pitch_A = np.eye(4)
res_roll_A = np.eye(4)
vt = -mass_ball*gravity/drag_coeff

ball_position_estimates = []
ball_time_estimates = []
projectile_computed = False

g_counter = 0
save_figure_logs = False
rospack = rospkg.RosPack()
# logging_file = open(rospack.get_path('shield_planner')+"/logs/Num_pts_projectile"+time.strftime("%Y%m%d_%H%M%S")+".txt", "w+")

def getProjectileDiscretizationPts(projectile_vel_x,
    projectile_vel_y,
    projectile_vel_z, 
    projectile_start_3d_x, 
    projectile_start_3d_y, 
    projectile_start_3d_z):

    projectile_start_x = projectile_start_3d_x
    projectile_start_y = projectile_start_3d_y
    projectile_start_z = projectile_start_3d_z

    time_interval = 0.08

    sx = []
    sy = []
    sz = []

    t = 0.0

    while t < 1.5:

        height_check = projectile_start_3d_z + projectile_vel_z*t - (1.0/2)*9.8*t*t
        if (height_check > -0.05):
            sx.append(projectile_start_3d_x + projectile_vel_x*t + (1.0/2)*0.0*t*t)
            sy.append(projectile_start_3d_y + projectile_vel_y*t + (1.0/2)*0.0*t*t)
            sz.append(height_check)

        t += time_interval

    print("sx ", sx)
    print("sy ", sy)
    print("sz ", sz)
    
    # range_3d = np.sqrt(np.pow((sx[0] - sz[-1]),2)
    range_3d = np.linalg.norm(np.array([sx[0], sy[0], sz[0]]) - np.array([sx[-1], sy[-1], sz[-1]]))

    print("range in 3d = ", range_3d)

    return sx, sy, sz

def getProjectileDiscretizationPts_2(projectile_vel_x,
    projectile_vel_y,
    projectile_vel_z, 
    projectile_start_3d_x, 
    projectile_start_3d_y, 
    projectile_start_3d_z, time_interval=0.01):
    global gravity, drag_coeff, mass_ball, vt

    g = -gravity

    projectile_start_x = projectile_start_3d_x
    projectile_start_y = projectile_start_3d_y
    projectile_start_z = projectile_start_3d_z

    sx = []
    sy = []
    sz = []

    t = 0.0

    while t < 1.5:

        height_check = projectile_start_z + (vt/g)*(projectile_vel_z+vt)*(1.0-np.exp(-(g*t/vt)))-vt*t
        if (height_check > -0.05):
            sx.append(projectile_start_3d_x + (projectile_vel_x*vt/g)*(1.0-np.exp(-g*t/vt)) )
            sy.append(projectile_start_3d_y + (projectile_vel_y*vt/g)*(1.0-np.exp(-g*t/vt)) )
            sz.append(height_check)

        t += time_interval
    range_3d = np.linalg.norm(np.array([sx[0], sy[0], sz[0]]) - np.array([sx[-1], sy[-1], sz[-1]]))
    return sx, sy, sz

def show_marker(sx, sy, sz, r=1.0, g=0.0,b=0.0):
    marker_array_ = MarkerArray()
    num_pts = len(sx)
    print("num_pts ", num_pts)
    for i in range(num_pts):
        marker_ = Marker()
        marker_.header.frame_id = "/odom_combined"
        # marker_.header.stamp = rospy.Time.now()
        marker_.type = marker_.SPHERE
        marker_.action = marker_.ADD

        marker_.pose.position.x = sx[i]
        marker_.pose.position.y = sy[i]
        marker_.pose.position.z = sz[i]
        marker_.pose.orientation.x = 0.0
        marker_.pose.orientation.y = 0.0
        marker_.pose.orientation.z = 0.0
        marker_.pose.orientation.w = 1.0
        marker_.id = i
        # marker_.lifetime = rospy.Duration.from_sec(lifetime_)
        marker_.scale.x = 0.05
        marker_.scale.y = 0.05
        marker_.scale.z = 0.05
        marker_.color.a = 0.8
        # red_, green_, blue_ = color_
        marker_.color.r = r
        marker_.color.g = g
        marker_.color.b = b
        marker_array_.markers.append(marker_) 
        
    return marker_array_

def compute_projectile_2():
    global ball_position_estimates, ball_time_estimates, projectile_time_msg_pub, projectile_marker_time_pub, vel_scale, gravity, vt
    s, e = 0, len(ball_position_estimates)-1
    vxm = []
    vym = []
    vzim = []

    for i in range(1,e+1):
        delx = ball_position_estimates[i][0] - ball_position_estimates[s][0]
        dely = ball_position_estimates[i][1] - ball_position_estimates[s][1]
        delz = ball_position_estimates[i][2] - ball_position_estimates[s][2]
        delt = ball_time_estimates[i] - ball_time_estimates[s]
        vx = (delx*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        vy = (dely*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        vzi = (((delz+vt*delt)*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))) - vt
        vxm.append(vx)
        vym.append(vy)
        vzim.append(vzi)
    vx = np.mean(vxm)
    vy = np.mean(vym)
    vzi = np.mean(vzim)

    projectile_msg = Projectile()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'odom_combined'
    projectile_msg.header = header

    projectile_msg.object_id = 0

    projectile_msg.position.x = ball_position_estimates[s][0]
    projectile_msg.position.y = ball_position_estimates[s][1]
    projectile_msg.position.z = ball_position_estimates[s][2]
    
    projectile_msg.velocity.x = vx
    projectile_msg.velocity.y = vy
    projectile_msg.velocity.z = vzi

    sx, sy, sz = getProjectileDiscretizationPts_2(vx, vy, vzi, 
        ball_position_estimates[s][0], ball_position_estimates[s][1], ball_position_estimates[s][2])
    marker_array = show_marker(sx, sy, sz, 1.0, 0.0, 0.0)

    projectile_time_msg_pub.publish(projectile_msg)
    projectile_marker_time_pub.publish(marker_array)

    # logging_file.write(str(header.stamp)+", "+str(len(ball_position_estimates))+'\n')
    print("projectile_msg ", projectile_msg)
    return projectile_msg


def compute_projectile():
    global R_SC_KI, T_SC_KI
    start_time = rospy.Time.now().to_sec()
    px_list = []
    for i in range(len(ball_position_estimates)):
        px_list.append(ball_position_estimates[i][0])
    px = np.array(px_list).reshape(len(ball_position_estimates), 1)

    py_list = []
    for i in range(len(ball_position_estimates)):
        py_list.append(ball_position_estimates[i][1])
    py = np.array(py_list).reshape(len(ball_position_estimates), 1)

    pz_list = []
    for i in range(len(ball_position_estimates)):
        pz_list.append(ball_position_estimates[i][2])
    pz = np.array(pz_list).reshape(len(ball_position_estimates), 1)
        

    range_x = [0, 10]
    range_y = [0, 10]
    range_z = [0, 10]

    parabola_coeffs = computeParabolaEqn(px, pz)

    # visualizeParabola3d(range_x, range_y, range_z, parabola_coeffs,
    #                     px, py, pz)
    
    # computing line equation formed by plane projection 
    m, c = computeLineEqn(px, py, plot_line=False) 

    # computing tranlation of new axis
    axis_translation_x = -c/m
    
    px_translated = px - axis_translation_x

    projectile_plane_angle = np.arctan2(m, 1)

    y_vel_direction = -1
    # if (projectile_plane_angle > 0.0):
    #     print("reversing y vel direction ")
    #     y_vel_direction = -1

    z_rot_matrix = getZRotMatrix(projectile_plane_angle)

    px_rot, py_rot, pz_rot = getTransformedPts(px_translated, py, pz, z_rot_matrix)    
    px_rot = px_rot + axis_translation_x

    parabola_x_z_plane_coeffs = computeParabolaEqn(px_rot, pz_rot)

    sqrt_tmp = np.abs(parabola_x_z_plane_coeffs[1]**2 - 4*parabola_x_z_plane_coeffs[0]*parabola_x_z_plane_coeffs[2])

    if (sqrt_tmp >= 0):
        solution_1 = (-parabola_x_z_plane_coeffs[1] + 
                        math.sqrt(sqrt_tmp)) / (2 * parabola_x_z_plane_coeffs[0])

        solution_2 = (-parabola_x_z_plane_coeffs[1] - 
                        math.sqrt(sqrt_tmp)) / (2 * parabola_x_z_plane_coeffs[0])
        if (solution_1 > solution_2):
            x_origin = solution_1
        else:
            x_origin = solution_2
    else:
        return False

    # print("x_origin ", x_origin)

    
    # m = 2*a*x_0 + b
    # projectile_slope = 2*parabola_x_z_plane_coeffs[0]*px_rot[0] + \
    #                     parabola_x_z_plane_coeffs[1]
    projectile_slope = 2*parabola_x_z_plane_coeffs[0]*x_origin + \
                    parabola_x_z_plane_coeffs[1]

    # theta = atan(m)
    projectile_angle = np.arctan2(projectile_slope, 1)

    visualizeParabola2d(range_x, range_y, range_z, px_rot, py_rot, pz_rot, 
                        parabola_x_z_plane_coeffs, projectile_slope)

    proj_range = computeProjectileRange(parabola_x_z_plane_coeffs)

    # x_vel, y_vel, z_vel = computeProjectileVel(proj_range, y_vel_direction, projectile_slope, projectile_plane_angle)
    x_vel, y_vel, z_vel = computeProjectileVelRInv(proj_range, y_vel_direction, projectile_angle, projectile_plane_angle, z_rot_matrix)

    x_origin = x_origin - axis_translation_x

    projectile_start_3d_x, projectile_start_3d_y, projectile_start_3d_z = np.dot(np.linalg.inv(z_rot_matrix), 
        np.array((x_origin, 0, 0)))

    projectile_start_3d_x = projectile_start_3d_x + axis_translation_x


    projectile_msg = Projectile()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'odom_combined'
    projectile_msg.header = header

    projectile_msg.object_id = 0

    projectile_msg.position.x = projectile_start_3d_x
    projectile_msg.position.y = projectile_start_3d_y
    projectile_msg.position.z = projectile_start_3d_z

    projectile_msg.velocity.x = x_vel*0.94
    projectile_msg.velocity.y = y_vel*0.94
    projectile_msg.velocity.z = z_vel*0.94

    end_time = rospy.Time.now().to_sec() - start_time

    sx, sy, sz = getProjectileDiscretizationPts(x_vel, y_vel, z_vel, 
        projectile_start_3d_x, projectile_start_3d_y, projectile_start_3d_z)
    marker_array = show_marker(sx, sy, sz, 0.0, 1.0, 0.0)

    projectile_msg_pub.publish(projectile_msg)
    projectile_marker_pub.publish(marker_array)

    rospy.loginfo("Finished compute_projectile_cb, returning now... ")
    rospy.loginfo("Time in py projectile compuation (msec) = %s", end_time*1000)

    # logging_file.write(str(header.stamp)+", "+str(len(ball_position_estimates))+'\n')
    print(projectile_msg)
    return projectile_msg

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
    global projectile_computed, A_FP_SC, ball_position_estimates, ball_time_estimates, x_offset, y_offset, z_offset
    t1 = rospy.Time.now().to_sec()
    global y_pos_lim, y_neg_lim, z_pos_lim, z_neg_lim, calibrate_frame, P_SC_B
    # if calibrate_frame>0:
    #     print("Calibrating")
    #     for data in pc2.read_points(ros_cloud, skip_nans=True):
    #         x, y, z = data[0], data[1] , data[2]
    #         if (z<z_pos_lim and z>0.05 and x>x_neg_lim and x<x_pos_lim):
    #             if y>0 and y<y_pos_lim:
    #                 y_pos_lim=y
    #             elif y<0 and y>y_neg_lim:
    #                 y_neg_lim=y
    #     calibrate_frame-=1
    #     if calibrate_frame==0:
    #         y_pos_lim = y_pos_lim*0.95
    #         y_neg_lim = y_neg_lim*0.95
    #         print("Calibration complete: Bounds", x_pos_lim, x_neg_lim, y_pos_lim, y_neg_lim, z_pos_lim, z_neg_lim)
    #     return

    pc = rnp.numpify(ros_cloud)
    points = np.zeros((pc.shape[0],3))
    points[:,0] = pc['x']
    points[:,1] = pc['y']
    points[:,2] = pc['z']

    # mask = np.concatenate([points[:,0:1]>x_neg_lim, points[:,0:1]<x_pos_lim, 
    #                        points[:,1:2]>y_neg_lim, points[:,1:2]<y_pos_lim, 
    #                        points[:,2:3]>z_neg_lim, points[:,2:3]<z_pos_lim], axis=1)
    # mask = np.all(mask, axis=1)

    ball_points = points[points[:,1]>y_neg_lim,:]
    ball_points = ball_points[ball_points[:,1]<y_pos_lim,:]
    # ball_points = ball_points[ball_points[:,1]>y_neg_lim,:]
    # ball_points = ball_points[ball_points[:,1]<y_pos_lim,:]
    ball_points = ball_points[ball_points[:,2]>z_neg_lim,:]
    ball_points = ball_points[ball_points[:,2]<z_pos_lim,:]

    t2 = rospy.Time.now().to_sec()
    
    if(ball_points.shape[0]<15 or ball_points.shape[0]>1500):
        return
    ball_position = np.mean(ball_points, axis=0)
    print("Ball points", ball_points.shape[0], ball_position)
    P_SC_B[0,0] = ball_position[0]
    P_SC_B[2,0] = ball_position[2]
    P_SC_B[1,0] = ball_position[1]
    # theta = 4.0*np.pi/180.0
    # res_A = np.array([[1,0,0,0],
    #                   [0,np.cos(theta),-np.sin(theta),0],
    #                   [0,np.sin(theta),np.cos(theta),0],
    #                   [0,0,0,1]])
    # A_FP_SC = np.dot(A_FP_SC,res_A)
    P_FP_B = np.dot(A_FP_SC, P_SC_B)
    ball_position_in_PR2 = P_FP_B[0:3,0]/P_FP_B[3,0]
    ball_position_in_PR2[2] += z_offset
    ball_position_in_PR2[1] += y_offset
    ball_position_in_PR2[0] += x_offset
    ball_position_estimates.append(ball_position_in_PR2)
    ball_time_estimates.append(ros_cloud.header.stamp.to_sec())

    if len(ball_position_estimates) >= 4 and not projectile_computed:
        #projectile_msg = compute_projectile()
        compute_projectile_2()
        # projectile_computed = True
        ball_position_estimates = []
        ball_time_estimates = []

    if G_DEBUG:
        dtype_list = rnp.point_cloud2.fields_to_dtype(ros_cloud.fields, ros_cloud.point_step)
        filtered_msg = xyzrgb_array_to_pointcloud2(ball_points,
                                                np.zeros(ball_points.shape, dtype=np.float32),
                                                ros_cloud.header.stamp,
                                                "launcher_camera",
                                                ros_cloud.header.seq)
        publisher.publish(filtered_msg)
    t3 = rospy.Time.now().to_sec()
    print(t3-t1,t2-t1, "loop time")



def callback(msg):
    filter_ball_points(msg)

    
def listener():
    global publisher
    global projectile_marker_pub
    global projectile_msg_pub, projectile_time_msg_pub, projectile_marker_time_pub
    global A_FP_SC, R_SC_KI, T_SC_KI, tf_trans, tf_rot
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    # A_FP_SC = np.eye(4)

    # listener = tf.TransformListener()
    tf_found = False
    # tf_trans = [5.574363423212862, -0.40031510516119606, 0.5773170124014723]
    # tf_rot = [0.44077743511951256, 0.43098939172865447, 0.5345096186509952, 0.5781547013234453]
    # tf_trans = [5.574363423212862, -0.40031510516119606, 0.5773170124014723]
    # tf_rot = [0.4655766643356737, 0.45389416488539236, 0.5152013737223944, 0.5583779664252736]
    # tf_trans = [5.574363423212862, -0.40031510516119606, 0.5773170124014723]
    # tf_rot = [0.5079512638580833, 0.49291718144432284, 0.47800103477546085, 0.5201279286033953]

    tf_trans = [6.01944448, -0.52128749,  0.55443587]
    tf_rot = [0.4995405,  0.45499593, 0.51035028, 0.53195919]

    # rate = rospy.Rate(10.0)
    # while not tf_found:
    #     try:
    #         (tf_trans,tf_rot) = listener.lookupTransform('base_footprint','launcher_camera',rospy.Time(0))
    #         print("translation ", tf_trans)
    #         print("rotation ", tf_rot)
    #         tf_found = True
    #     except:
    #         print("TF not found")
    #     rate.sleep()
    # print("Found TF")
    trans_mat = tf.transformations.translation_matrix(tf_trans)
    rot_mat = tf.transformations.quaternion_matrix(tf_rot)
    A_FP_SC = np.dot(trans_mat, rot_mat)

    global x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset, drag_coeff, vt, res_roll_A, res_pitch_A, res_yaw_A, mass_ball
    params = tuple(np.load("/home/roman/catkin_ws/perpection_params.npy").tolist())
    print("Params loaded", params)
    x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset, _ = params
    drag_coeff = 0.015
    pitch_offset, roll_offset, yaw_offset = pitch_offset*np.pi/180.0, roll_offset*np.pi/180.0, yaw_offset*np.pi/180.0
    vt = -mass_ball*gravity/drag_coeff
    res_pitch_A = np.array([[1,0,0,0],
                      [0,np.cos(pitch_offset),-np.sin(pitch_offset),0],
                      [0,np.sin(pitch_offset),np.cos(pitch_offset),0],
                      [0,0,0,1]])
    res_yaw_A = np.array([[np.cos(yaw_offset),0,np.sin(yaw_offset),0],
                      [0,1.0,0,0],
                      [-np.sin(yaw_offset),0,np.cos(yaw_offset),0],
                      [0,0,0,1]])
    res_roll_A = np.array([[np.cos(roll_offset),-np.sin(roll_offset),0,0],
                      [np.sin(roll_offset),np.cos(roll_offset),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    A_res = np.dot(res_pitch_A,res_yaw_A)
    A_res = np.dot(res_roll_A,A_res)
    A_FP_SC = np.dot(A_FP_SC,A_res)
    print("Transformation Loaded", A_FP_SC)

    rospy.Subscriber("/passthrough/output", PointCloud2, callback)
    publisher = rospy.Publisher('/sc/rgbd/filtered_points', PointCloud2, queue_size=1)
    projectile_msg_pub = rospy.Publisher("projectile_3", Projectile, queue_size=1)
    projectile_marker_pub = rospy.Publisher("projectile_3_vis", MarkerArray, queue_size=10)

    projectile_time_msg_pub = rospy.Publisher("projectile", Projectile, queue_size=1)
    projectile_marker_time_pub = rospy.Publisher("projectile_vis", MarkerArray, queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()