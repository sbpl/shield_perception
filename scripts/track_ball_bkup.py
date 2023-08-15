#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time
import pickle
import copy

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

from dynamic_reconfigure.server import Server
# from shield_perception.cfg import calibrateConfig

## Config Variables
G_DEBUG = True
x_cen = -0.4515569
x_width = 0.768414*0.8
y_cen = -0.0709
y_width = 1.2218*0.97
z_pos_lim = 3.
z_neg_lim = 0.01
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

## Runtime Variables
publisher, k_publisher = None, None
projectile_marker_pub, k_projectile_marker_pub = None, None
projectile_msg_pub = None
x_pos_lim = x_cen + x_width/2.0
x_neg_lim = x_cen - x_width/2.0
y_pos_lim = y_cen + y_width/2.0
y_neg_lim = y_cen - y_width/2.0
A_FP_SC = None
A_FP_KI = None
P_SC_B = np.ones((4,1))
res_yaw_A = np.eye(4)
res_pitch_A = np.eye(4)
res_roll_A = np.eye(4)
vt = -mass_ball*gravity/drag_coeff
ball_position_estimates = []
ball_time_estimates = []
ball_position_estimates_base = []
ball_pointcloud = []
projectile_computed = False
k_ball_position_estimates = []
k_ball_pointcloud = []
k_ball_time_estimates = []
k_projectile_computed = False
k_enable_replanning = True
k_kinect_replanning_available = False
projectile_estimates = []
first_sc_timestamp = None


# def getProjectileDiscretizationPts_2(projectile_vel_x,
#     projectile_vel_y,
#     projectile_vel_z, 
#     projectile_start_3d_x, 
#     projectile_start_3d_y, 
#     projectile_start_3d_z, time_interval=0.01):
#     global gravity, drag_coeff, mass_ball, vt
#
#     g = -gravity
#
#     projectile_start_x = projectile_start_3d_x
#     projectile_start_y = projectile_start_3d_y
#     projectile_start_z = projectile_start_3d_z
#
#     sx = []
#     sy = []
#     sz = []
#
#     t = 0.0
#
#     while t < 1.5:
#
#         height_check = projectile_start_z + (vt/g)*(projectile_vel_z+vt)*(1.0-np.exp(-(g*t/vt)))-vt*t
#         if (height_check > -0.05):
#             sx.append(projectile_start_3d_x + (projectile_vel_x*vt/g)*(1.0-np.exp(-g*t/vt)) )
#             sy.append(projectile_start_3d_y + (projectile_vel_y*vt/g)*(1.0-np.exp(-g*t/vt)) )
#             sz.append(height_check)
#
#         t += time_interval
#
#     # print("sx ", sx)
#     # print("sy ", sy)
#     # print("sz ", sz)
#     
#     # range_3d = np.sqrt(np.pow((sx[0] - sz[-1]),2)
#     range_3d = np.linalg.norm(np.array([sx[0], sy[0], sz[0]]) - np.array([sx[-1], sy[-1], sz[-1]]))
#
#     # print("range in 3d = ", range_3d)
#
#     return sx, sy, sz


# def show_marker(sx, sy, sz, r=1.0, g=0.0,b=0.0):
#     marker_array_ = MarkerArray()
#     num_pts = len(sx)
#     # print("num_pts ", num_pts)
#     for i in range(num_pts):
#         marker_ = Marker()
#         marker_.header.frame_id = "/odom_combined"
#         # marker_.header.stamp = rospy.Time.now()
#         marker_.type = marker_.SPHERE
#         marker_.action = marker_.ADD
#
#         marker_.pose.position.x = sx[i]
#         marker_.pose.position.y = sy[i]
#         marker_.pose.position.z = sz[i]
#         marker_.pose.orientation.x = 0.0
#         marker_.pose.orientation.y = 0.0
#         marker_.pose.orientation.z = 0.0
#         marker_.pose.orientation.w = 1.0
#         marker_.id = i
#         # marker_.lifetime = rospy.Duration.from_sec(lifetime_)
#         marker_.scale.x = 0.05
#         marker_.scale.y = 0.05
#         marker_.scale.z = 0.05
#         marker_.color.a = 0.8
#         # red_, green_, blue_ = color_
#         marker_.color.r = r
#         marker_.color.g = g
#         marker_.color.b = b
#         marker_array_.markers.append(marker_) 
#         
#     return marker_array_

def compute_projectile_2(bpes, btes, pmsg, pmmsg, color=(1.0,0.0,0.0), publish_pmsg=True):
    global vel_scale, gravity, vt, projectile_estimates
    t1 = rospy.Time.now()
    s, e = 0, 2 ## SC will always send 3
    vxm = []
    vym = []
    vzim = []

    g = -gravity

    if len(bpes) > 3:
        # for i in [e]:# range(1,e+1):
        #     delx = bpes[i][0] - bpes[s][0]
        #     dely = bpes[i][1] - bpes[s][1]
        #     delz = bpes[i][2] - bpes[s][2]
        #     delt = btes[i] - btes[s]
        #     # vx = delx/delt
        #     # vy = dely/delt
        #     # vzi = (delz-0.5*gravity*delt*delt)/delt
        #     vx = (delx*g)/(vt*(1.0-np.exp(-g*delt/vt)))
        #     vy = (dely*g  )/(vt*(1.0-np.exp(-g*delt/vt)))
        #     vzi = (((delz+vt*delt)*g)/(vt*(1.0-np.exp(-g*delt/vt)))) - vt
        #     vxm.append(vx)
        #     vym.append(vy)
        #     vzim.append(vzi)

        print("Re planning")
        e = len(bpes)-1
        print(bpes[e], btes, bpes[s], "BPES")
        for i in range(2,e+1):
            delx = bpes[i][0] - bpes[s][0]
            dely = bpes[i][1] - bpes[s][1]
            delz = bpes[i][2] - bpes[s][2]
            delt = btes[i] - btes[s]
            # vx = delx/delt
            # vy = dely/delt
            # vzi = (delz-0.5*gravity*delt*delt)/delt
            vx = (delx*g)/(vt*(1.0-np.exp(-g*delt/vt)))
            vy = (dely*g  )/(vt*(1.0-np.exp(-g*delt/vt)))
            vzi = (((delz+vt*delt)*g)/(vt*(1.0-np.exp(-g*delt/vt)))) - vt
            vxm.append(vx)
            vym.append(vy)
            vzim.append(vzi)
        print("============ Replanned after", btes[3]-btes[2], " timestamp=", btes[3])
    else:
        print("First")
        for i in [e]:
            delx = bpes[i][0] - bpes[s][0]
            dely = bpes[i][1] - bpes[s][1]
            delz = bpes[i][2] - bpes[s][2]
            delt = btes[i] - btes[s]
            # vx = delx/delt
            # vy = dely/delt
            # vzi = (delz-0.5*gravity*delt*delt)/delt
            vx = (delx*g)/(vt*(1.0-np.exp(-g*delt/vt)))
            vy = (dely*g  )/(vt*(1.0-np.exp(-g*delt/vt)))
            vzi = (((delz+vt*delt)*g)/(vt*(1.0-np.exp(-g*delt/vt)))) - vt
            vxm.append(vx)
            vym.append(vy)
            vzim.append(vzi)
    vx = np.mean(vxm)
    vy = np.mean(vym)
    vzi = np.mean(vzim)

    # print("Vels:",vxm,vym,vzim,np.mean(np.subtract(vxm[1:],vxm[0:-1])), np.mean(np.subtract(vym[1:],vym[0:-1])), np.mean(np.subtract(vzim[1:],vzim[0:-1])) )

    projectile_msg = Projectile()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'odom_combined'
    projectile_msg.header = header

    projectile_msg.object_id = 0

    if len(bpes) <= 3 and publish_pmsg:
        errorz = 0.0
        print("Introducing Error of ", errorz)
        bpes[s][1] += errorz

    projectile_msg.position.x = bpes[s][0]
    projectile_msg.position.y = bpes[s][1]
    projectile_msg.position.z = bpes[s][2]
    
    projectile_msg.velocity.x = vx
    projectile_msg.velocity.y = vy
    projectile_msg.velocity.z = vzi

    projectile_estimates.append([bpes[s][0], bpes[s][1], bpes[s][2], vx, vy, vzi])
    print("Compute Projectile:", [bpes[s][0], bpes[s][1], bpes[s][2], vx, vy, vzi])

    # sx, sy, sz = getProjectileDiscretizationPts_2(vx, vy, vzi, 
    #     bpes[s][0], bpes[s][1], bpes[s][2])
    # marker_array = show_marker(sx, sy, sz, color[0], color[1], color[2])
    if publish_pmsg:
        pmsg.publish(projectile_msg)
        print(projectile_msg)
    # pmmsg.publish(marker_array)
    # print(projectile_msg)
    t2 = rospy.Time.now()
    print("Diff", (t2-t1).to_sec()*1000, t2)
    return projectile_msg

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

def extrapolate_projectile_to_time(px,py,pz,vx,vy,vz,delt):
    global gravity, vt 
    g = -gravity
    ez = pz + (vt/g)*(vz+vt)*(1.0-np.exp(-(g*delt/vt)))-vt*delt
    ex = px + (vx*vt/g)*(1.0-np.exp(-g*delt/vt))
    ey = py + (vy*vt/g)*(1.0-np.exp(-g*delt/vt))
    return np.array([ex,ey,ez])

def compute_accuracy(ipx,ipy,ipz,ivx,ivy,ivz,wall_x=1.0):
    ki_points = np.array(k_ball_position_estimates).transpose()
    ki_times = np.array(k_ball_time_estimates)-first_sc_timestamp
    ## Compute velocity of GT projectile
    s, e = 0, ki_points.shape[1]-1
    k_vxm = []
    k_vym = []
    k_vzim = []
    for i in range(1,e+1):
        delx = ki_points[0,i] - ki_points[0,s]
        dely = ki_points[1,i] - ki_points[1,s]
        delz = ki_points[2,i] - ki_points[2,s]
        delt = ki_times[i] - ki_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        k_vxm.append(vx)
        k_vym.append(vy)
        k_vzim.append(vzi)
    k_vx = np.mean(k_vxm)
    k_vy = np.mean(k_vym)
    k_vzi = np.mean(k_vzim)
    
    ki_delx = wall_x-ki_points[0,0]
    ki_delt = ki_delx/(k_vx)
    ki_point = extrapolate_projectile_to_time( ki_points[0,0],ki_points[1,0],ki_points[2,0],k_vx,k_vy,k_vzi,ki_delt)
    reference_points = np.concatenate([ki_points, ki_point.reshape(-1,1)], axis=1)

    extrapolated_points = np.array([extrapolate_projectile_to_time( ipx,ipy,ipz,ivx,ivy,ivz,delt ) for delt in np.arange(ki_times[0]-0.030, ki_delt+ki_times[0], 0.005)])
    extrapolated_points = extrapolated_points.transpose()

    if (extrapolated_points.shape[0]==0):
        print("Error Extrapolating Points")

    diff_list = []
    for i in range(reference_points.shape[1]):
        diff = np.min(np.linalg.norm(extrapolated_points-reference_points[:,i:i+1], axis=0))
        diff_list.append(diff)

    error = np.mean(diff_list)
    return error
    


def filter_ball_points(ros_cloud):
    global projectile_computed, ball_position_estimates_base, ball_time_estimates, ball_pointcloud
    global y_pos_lim, y_neg_lim, z_pos_lim, z_neg_lim
    if projectile_computed:
        return

    point_list = []
    pc = rnp.numpify(ros_cloud)
    points = np.zeros((pc.shape[0],3))
    points[:,0] = pc['x']
    points[:,1] = pc['y']
    points[:,2] = pc['z']

    ball_points = points[points[:,1]>y_neg_lim,:]
    ball_points = ball_points[ball_points[:,1]<y_pos_lim,:]
    ball_points = ball_points[ball_points[:,2]>z_neg_lim,:]
    ball_points = ball_points[ball_points[:,2]<z_pos_lim,:]
    
    if(ball_points.shape[0]<15):
        return
    ball_position = np.mean(ball_points, axis=0)

    ball_position_estimates_base.append(ball_position)
    ball_pointcloud.append(ball_points)
    ball_time_estimates.append(ros_cloud.header.stamp.to_sec())
    print("Structure Core Ball points", ball_points.shape[0], ball_position, ros_cloud.header.stamp.to_sec())

    if len(ball_position_estimates_base) >= 5 and not projectile_computed:
        print("=============STRUCTURE CORE===========================", rospy.Time.now().to_sec())
        projectile_computed = True

    if G_DEBUG:
        dtype_list = rnp.point_cloud2.fields_to_dtype(ros_cloud.fields, ros_cloud.point_step)
        filtered_msg = xyzrgb_array_to_pointcloud2(ball_points,
                                                np.zeros(ball_points.shape, dtype=np.float32),
                                                ros_cloud.header.stamp,
                                                "launcher_camera",
                                                ros_cloud.header.seq)
        publisher.publish(filtered_msg)
    t3 = rospy.Time.now().to_sec()
    # print(t3-t1,t2-t1, "loop time")

def ros_to_pcl(ros_cloud):
    points_list = []
    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2]])
    return np.array(points_list) 

def kinect_callback(ros_cloud):
    global k_projectile_computed, k_ball_position_estimates, k_ball_time_estimates, A_FP_KI, k_ball_pointcloud, k_kinect_replanning_available
    if k_projectile_computed or not projectile_computed:
        return

    ball_points = ros_to_pcl(ros_cloud)

    if len(k_ball_position_estimates)==0 and ball_points.shape[0]<1:
        return
    elif ball_points.shape[0]<50:
        return
    
    # if(ball_points.shape[0]<50):
    #     return

    ball_position = np.mean(ball_points, axis=0)
    hom_ball_position = np.ones((4,1))
    hom_ball_position[0,0] = ball_position[0]
    hom_ball_position[2,0] = ball_position[2]
    hom_ball_position[1,0] = ball_position[1]
    P_FP_B = np.dot(A_FP_KI, hom_ball_position)
    ball_position_in_PR2 = P_FP_B[0:3,0]/P_FP_B[3,0]
    print("kinect Ball points", ball_points.shape[0], ball_position_in_PR2, ros_cloud.header.stamp.to_sec())

    if projectile_computed:
        k_ball_position_estimates.append(ball_position_in_PR2)
        k_ball_time_estimates.append(ros_cloud.header.stamp.to_sec())
        k_ball_pointcloud.append(ball_points)
    
    if len(k_ball_position_estimates) >= 1 and not k_projectile_computed:
        k_kinect_replanning_available = True

    if len(k_ball_position_estimates) >= 3 and not k_projectile_computed:
        print("==============kinect==========================")
        if projectile_computed:
            save_dict = {}
            save_dict['sc'] = (ball_position_estimates_base, ball_time_estimates)
            save_dict['sc_points'] = ball_pointcloud
            save_dict['ki'] = (k_ball_position_estimates, k_ball_time_estimates)
            save_dict['ki_points'] = k_ball_pointcloud
            save_dict['TF'] = A_FP_KI
            rospack = rospkg.RosPack()
            pkg_dir = rospack.get_path('shield_perception')
            filename = pkg_dir + "/data_dump/calib_data_runs/"+time.strftime("%Y%m%d-%H%M%S")+'.pickle' 
            with open(filename, 'wb') as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Saving raw files to ", filename)
            for estimates in projectile_estimates:
                # print(estimates, "Estimates")
                print(compute_accuracy(estimates[0],estimates[1],estimates[2],estimates[3],estimates[4],estimates[5])*100.0, "Errors (cm)")
                print("Time Diff (s) = ", k_ball_time_estimates[0]-ball_time_estimates[-1])
        k_projectile_computed = True

    # if G_DEBUG:
    dtype_list = rnp.point_cloud2.fields_to_dtype(ros_cloud.fields, ros_cloud.point_step)
    filtered_msg = xyzrgb_array_to_pointcloud2(ball_points,
                                            np.zeros(ball_points.shape, dtype=np.float32),
                                            ros_cloud.header.stamp,
                                            ros_cloud.header.frame_id,
                                            ros_cloud.header.seq)
    k_publisher.publish(filtered_msg)

def listener():
    global publisher, k_publisher
    global projectile_msg_pub, projectile_marker_pub, k_projectile_marker_pub
    global A_FP_SC, A_FP_KI
    rospy.init_node('c_p_calibrate')
    tf_found = False

    rate = rospy.Rate(10.0)
    k_tf_trans = None
    k_tf_rot = None
    tf_listener = tf.TransformListener()
    # Find tf (transform) listener
    while not tf_found:
        try:
            # Look for trans and rot? for camera_depth_optical_frame
            # frome base_footprint
            (k_tf_trans,k_tf_rot) = tf_listener.lookupTransform("base_footprint",\
                    "/camera_depth_optical_frame",rospy.Time(0))
            print("translation ", k_tf_trans)
            print("rotation ", k_tf_rot)
            tf_found = True
        except Exception as e:
            print("TF not found", e)
        rate.sleep()
    print("Found TF")
    # k_tf_trans = [-0.15783123802738946, 0.01273613573344717, 1.4690869133055597]
    # k_tf_rot = [-0.4749945606732899, 0.5031187344211056, -0.5249774084103889, 0.4956313418903257]
    # Multiplying trans and rot to get relative to base_footprint
    trans_mat = tf.transformations.translation_matrix(k_tf_trans)
    rot_mat = tf.transformations.quaternion_matrix(k_tf_rot)
    A_FP_KI = np.dot(trans_mat, rot_mat)

    # Const trans and rot for ?
    tf_trans = [6.01944448, -0.52128749,  0.55443587]
    tf_rot = [0.4995405,  0.45499593, 0.51035028, 0.53195919]

    trans_mat = tf.transformations.translation_matrix(tf_trans)
    rot_mat = tf.transformations.quaternion_matrix(tf_rot)
    A_FP_SC = np.dot(trans_mat, rot_mat)
    print("Transformation Loaded", A_FP_SC)

    rospy.Subscriber("/kinect_filtered/output", PointCloud2, kinect_callback)
    rospy.Subscriber("/external_filtered/output", PointCloud2, filter_ball_points)
    publisher = rospy.Publisher('/sc/rgbd/filtered_points', PointCloud2, queue_size=1)
    k_publisher = rospy.Publisher('/kinect/filtered_points', PointCloud2, queue_size=1)

    projectile_msg_pub = rospy.Publisher("projectile", Projectile, queue_size=1)
    projectile_marker_pub = rospy.Publisher("projectile_vis", MarkerArray, queue_size=10)
    k_projectile_marker_pub = rospy.Publisher("kinect_projectile_vis", MarkerArray, queue_size=10)

    # srv = Server(calibrateConfig, dyn_reconf_cb)
    global x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset, drag_coeff, vt, res_roll_A, res_pitch_A, res_yaw_A, mass_ball
    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path('shield_perception')
    param_filename = pkg_dir + "/data_dump/perception_params.npy"
    params = tuple(np.load(param_filename).tolist())
    print("Param file ", param_filename)
    # params = [0.404571, 0.13428446, 0.35198332, 1.5798978885211274, -8.995515878771856, -0.34771949149796966, 0.01176908]
    print("Params loaded", params)
    x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset, drag_coeff = params
    # drag_coeff = 0.01176908 
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

    rate = rospy.Rate(600) # 10hz
    global projectile_computed, ball_position_estimates, ball_position_estimates_base, res_yaw_A, res_pitch_A, z_offset, x_offset, y_offset, P_SC_B
    global k_enable_replanning, first_sc_timestamp
    process_sc = True
    while not rospy.is_shutdown():
        if projectile_computed and process_sc:
            t1 = rospy.Time.now()
            ball_position_estimates_base_filtered = []
            ball_time_estimates_filtered = []
            errors = []
            for i in range(5):
                error = np.max(np.linalg.norm(np.array(ball_pointcloud[i])-ball_position_estimates_base[i].reshape((1,-1)), axis=1))
                errors.append(error)
            order = np.argsort(errors)
            print("======== Order======== ", order)
            for i in range(5):
                if i in order[0:3]:
                    ball_position_estimates_base_filtered.append(ball_position_estimates_base[i])
                    ball_time_estimates_filtered.append(ball_time_estimates[i])

            sc_points = np.concatenate([np.array(ball_position_estimates_base_filtered).transpose(), np.ones((1,len(ball_position_estimates_base_filtered)))], axis=0)
            pr2_frame_points = np.linalg.multi_dot([A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
            pr2_frame_points += np.array([[x_offset],[y_offset],[z_offset],[0.0]])
            ball_position_estimates = pr2_frame_points[0:3,:].transpose().tolist()
            t2 = rospy.Time.now()
            compute_projectile_2(ball_position_estimates, ball_time_estimates_filtered, projectile_msg_pub, projectile_marker_pub)
            process_sc = False
            print("Structure CORE TimeSTAMPS", ball_time_estimates, "time diff", (t2-t1).to_sec()*1000, t1.to_sec())
        if k_enable_replanning and k_kinect_replanning_available:
            ball_position_estimates_base_filtered = []
            ball_time_estimates_filtered = []
            errors = []
            for i in range(5):
                error = np.max(np.linalg.norm(np.array(ball_pointcloud[i])-ball_position_estimates_base[i].reshape((1,-1)), axis=1))
                errors.append(error)
            order = np.argsort(errors)
            print("======== Order======== ", order)
            for i in range(5):
                if i in order[0:3]:
                    ball_position_estimates_base_filtered.append(ball_position_estimates_base[i])
                    ball_time_estimates_filtered.append(ball_time_estimates[i])

            sc_points = np.concatenate([np.array(ball_position_estimates_base_filtered).transpose(), np.ones((1,len(ball_position_estimates_base_filtered)))], axis=0)
            pr2_frame_points = np.linalg.multi_dot([A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
            pr2_frame_points += np.array([[x_offset],[y_offset],[z_offset],[0.0]])
            first_sc_timestamp = ball_time_estimates_filtered[0]
            ball_position_estimates = pr2_frame_points[0:3,:].transpose().tolist()
            compute_projectile_2(ball_position_estimates+np.array(k_ball_position_estimates).tolist(), 
                                 ball_time_estimates_filtered+k_ball_time_estimates,
                                 projectile_msg_pub, projectile_marker_pub)
            k_enable_replanning = False
        if k_projectile_computed:
            # compute_projectile_2(k_ball_position_estimates, k_ball_time_estimates, projectile_msg_pub, k_projectile_marker_pub, color=(0.0,1.0,0.0), publish_pmsg=False)
            break
        rate.sleep()

if __name__ == '__main__':
    listener()
