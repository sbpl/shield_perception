#!/usr/bin/env python
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time
import pickle

# projectile imports
from shield_planner_msgs.srv import TestbedProjectile
from shield_planner_msgs.msg import Projectile
import std_msgs.msg
import rospkg
from visualization_msgs.msg import MarkerArray, Marker
import tf

from dynamic_reconfigure.server import Server

import pdb
# from shield_perception.cfg import calibrateConfig

# Macro: 
# METHOD_ID: 0 = bounding box, 1 = color filtering
# NUM_FRAME: number of frames required to start estimation
# COLOR_FILTER: sets of color filtering in pc
## 0: nothing
## 1: lower = (160, 40, 40) upper = (250, 125, 125) - working, well?
# PRINT_COLOR: print out color's bound
# OUTLIER_REJECT: perform outlier reject or not
METHOD_ID=0
NUM_FRAME=3
COLOR_FILTER=0
PRINT_COLOR=0
OUTLIER_REJECT=1

##########################################################################
#############################Function#####################################
# Helper function to turn xyzrgb array to pc2 data structure
def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    # Process args
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    else:
        N = points.shape[0]
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N
    # Setting up message's fields and opitions
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = xyzrgb.tobytes()
    return msg

# Helper function for ros to point cloud list
def ros_to_pcl(ros_cloud):
    points_list = []
    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2]])
    return np.array(points_list)


# Call back function for structure core
def structure_callback( ros_cloud, args ):
    track_obj = args[0]
    sc_obj = args[1]
    # ki_obj = args[2]
    # Getting spatial limit from track ball object
    # y_pos_lim = track_obj.y_pos_lim
    # y_neg_lim = track_obj.y_neg_lim
    # z_pos_lim = track_obj.z_pos_lim
    # z_neg_lim = track_obj.z_neg_lim
    # If projectile already computed, just return
    if sc_obj.projectile_computed:
        return
    # Converting point cloud to python vectors
    point_list = []
    pc = rnp.numpify(ros_cloud)
    points = np.zeros((pc.shape[0],3),dtype=np.float32)
    points_rgb = np.zeros((pc.shape[0],1),dtype=np.float32)
    points[:,0] = pc['x']
    points[:,1] = pc['y']
    points[:,2] = pc['z']
    if 'rgb' in pc.dtype.fields:
        points_rgb = pc['rgb']
        pc_rgb = rnp.point_cloud2.split_rgb_field(pc)

        if pc_rgb['r'].shape[0] > 0:
            # This part is to find the bound for pc rgb filtering\
            if PRINT_COLOR:
                r_min = np.min(pc_rgb['r'])
                g_min = np.min(pc_rgb['g'])
                b_min = np.min(pc_rgb['b'])
                r_max = np.max(pc_rgb['r'])
                g_max = np.max(pc_rgb['g'])
                b_max = np.max(pc_rgb['b'])
                print("The RGB lower bound is:({} {} {})".format(r_min, g_min, b_min))
                print("The RGB upper bound is:({} {} {})".format(r_max, g_max, b_max))
        
            # Color filtering
            # pdb.set_trace()
            if COLOR_FILTER==1:
                # Finding common indices
                ind_r = np.intersect1d(np.where(pc_rgb['r'] > 160)[0], np.where(pc_rgb['r'] < 250)[0])
                ind_g = np.intersect1d(np.where(pc_rgb['g'] > 40)[0], np.where(pc_rgb['g'] < 125)[0])
                ind_b = np.intersect1d(np.where(pc_rgb['b'] > 40)[0], np.where(pc_rgb['b'] < 125)[0])
                ind = np.intersect1d(np.intersect1d(ind_r, ind_g), ind_b)
                points = points[ind,:]
                points_rgb = points_rgb[ind]
            if COLOR_FILTER==2:
                # Finding common indices
                ind_r = np.intersect1d(np.where(pc_rgb['r'] > 150)[0], np.where(pc_rgb['r'] < 250)[0])
                ind_g = np.intersect1d(np.where(pc_rgb['g'] > 50)[0], np.where(pc_rgb['g'] < 100)[0])
                ind_b = np.intersect1d(np.where(pc_rgb['b'] > 40)[0], np.where(pc_rgb['b'] < 100)[0])
                ind = np.intersect1d(np.intersect1d(ind_r, ind_g), ind_b)
                points = points[ind,:]
                points_rgb = points_rgb[ind]

    # pdb.set_trace()
    # Filter out some points that exceed spatial limit
    # ball_points = points[points[:,1]>y_neg_lim,:]
    # ball_points = ball_points[ball_points[:,1]<y_pos_lim,:]
    # ball_points = ball_points[ball_points[:,2]>z_neg_lim,:]
    # ball_points = ball_points[ball_points[:,2]<z_pos_lim,:]
    # Only count the points when sc capture enough (15+) points
    if(points.shape[0]<15):
        return
    
    # Outlier rejection
    if OUTLIER_REJECT:
        x_mean = np.mean(points[:,0])
        y_mean = np.mean(points[:,1])
        z_mean = np.mean(points[:,2])
        x_std = np.std(points[:,0])
        y_std = np.std(points[:,1])
        z_std = np.std(points[:,2])
        ind_x = np.intersect1d(np.where(points[:,0] > x_mean - 2*x_std)[0], np.where(points[:,0] < x_mean + 2*x_std)[0])
        ind_y = np.intersect1d(np.where(points[:,1] > y_mean - 2*y_std)[0], np.where(points[:,1] < y_mean + 2*y_std)[0])
        ind_z = np.intersect1d(np.where(points[:,2] > z_mean - 2*z_std)[0], np.where(points[:,2] < z_mean + 2*z_std)[0])
        ind = np.intersect1d(np.intersect1d(ind_x, ind_y), ind_z)
        points = points[ind,:]
        points_rgb = points_rgb[ind]
        

    # Transform the pointcloud into base_link frame
    sc_points = np.concatenate(
                  [points.transpose(),
                  np.ones((1,points.shape[0]))],
                  axis=0)
    base_frame_points = np.linalg.multi_dot(
                    [sc_obj.A_FP_SC, track_obj.res_roll_A,
                    track_obj.res_pitch_A, track_obj.res_yaw_A,
                    sc_points])
    points = base_frame_points[0:3,:].transpose()
    ball_position = np.mean(points, axis=0)
    # Save data into data structure in object
    sc_obj.ball_time_estimates.append(ros_cloud.header.stamp.to_sec())
    sc_obj.ball_position_estimates_base.append(ball_position)
    sc_obj.ball_pointcloud.append(points)
    sc_obj.ball_pointcloud_rgb.append(points_rgb)
    # Print out data for debugging purpose
    print("Structure Core Ball points",
          points.shape[0], ball_position,
          ros_cloud.header.stamp.to_sec())
    # If collected more than 5 points, mark projectile as computed
    # for structure core
    if len(sc_obj.ball_position_estimates_base) >= NUM_FRAME and not\
            sc_obj.projectile_computed:
        print("=============STRUCTURE CORE===========================",
              rospy.Time.now().to_sec())
        sc_obj.projectile_computed = True
    # If debug mode is set, publish data collected to filterd_msg
    # if track_obj.g_debug:
    #     dtype_list = rnp.point_cloud2.fields_to_dtype(
    #                ros_cloud.fields,
    #                ros_cloud.point_step )
    #     filtered_msg = xyzrgb_array_to_pointcloud2(
    #                  points,
    #                  np.zeros(points.shape, dtype=np.float64),
    #                  ros_cloud.header.stamp,
    #                  "launcher_camera",
    #                  ros_cloud.header.seq )
    #     sc_obj.publisher.publish(filtered_msg)
    # Timer for loop time
    # t3 = rospy.Time.now().to_sec()
    # print(t3-t1,t2-t1, "loop time")

##########################################################################
#############################Classes######################################
class TrackBall( object ) :
    '''
    Session to track ball projectile in Shield Project
    '''
    def __init__( self ):
        # Config
        self.g_debug = False

        # Spatial
        # self.x_cen = -0.4515569
        # self.x_width = 0.768414*0.8
        # self.y_cen = -0.0709
        # self.y_width = 1.2218*0.97
        # self.z_pos_lim = 3.
        # self.z_neg_lim = 0.01
        # self.vel_scale = 1.0 # Not used 
        self.z_offset = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.roll_offset = 0.0
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        # self.x_pos_lim = self.x_cen + self.x_width/2.0
        # self.x_neg_lim = self.x_cen - self.x_width/2.0
        # self.y_pos_lim = self.y_cen + self.y_width/2.0
        # self.y_neg_lim = self.y_cen - self.y_width/2.0
        self.res_yaw_A = np.eye(4)
        self.res_pitch_A = np.eye(4)
        self.res_roll_A = np.eye(4)

        # Projectile Model
        self.gravity = -9.80665
        self.drag_coeff = 0.015
        self.mass_ball = 0.03135
        self.vt = -self.mass_ball*self.gravity/self.drag_coeff

        # Data
        self.projectile_estimates = []

        # Create Kinect and structure core session
        # self.ki_sess = KISession()
        self.sc_sess = SCSession()

    # Function to publish points used to estimate the projectile
    def visualize_projectile_points( self, bpes, pmmsg ):
        markers = MarkerArray()
        for ind, es in enumerate(bpes):
            marker = Marker()
            marker.header.frame_id = "odom_combined"
            marker.type = marker.SPHERE
            marker.id = ind
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 0.3
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = es[0]
            marker.pose.position.y = es[1] 
            marker.pose.position.z = es[2]
            markers.markers.append(marker)
        pmmsg.publish(markers)

    # Function to visualized pc2 that are collected
    def visualize_collected_pc( self, pc_list, pc_rgb_list, pub ):
        pc_all = np.zeros((0,3))
        pc_rgb_all = np.zeros((0,1))
        for i in range(len(pc_list)):
            pc_all = np.vstack((pc_all, np.array(pc_list[i])))            
            # pdb.set_trace()
            pc_rgb_all = np.vstack((pc_rgb_all, np.array(pc_rgb_list[i]).reshape(-1,1)))

        # Publishing
        filtered_msg = xyzrgb_array_to_pointcloud2(
                    pc_all,
                    pc_rgb_all,
                    rospy.Time.now(),
                    "odom_combined")
        pub.publish(filtered_msg)

    # Helper function to compute projectile
    def compute_projectile_2(
            self, bpes, btes, pmsg, pmmsg,
            color=(1.0,0.0,0.0),
            publish_pmsg=True
            ):
        ''' Compute projectile function in Track ball '''
        # Local variable setup
        t1 = rospy.Time.now()
        s, e = 0, 2 ## SC will always send 3
        vxm = []
        vym = []
        vzim = []
        g = -self.gravity
        vt = self.vt

        # Compute projectile
        if len(bpes) > 3:
            print("Replanning")
            e = len(bpes)-1
            print(bpes[e], btes, bpes[s], "BPES")
            for i in range(2,e+1):
                delx = bpes[i][0] - bpes[s][0]
                dely = bpes[i][1] - bpes[s][1]
                delz = bpes[i][2] - bpes[s][2]
                delt = btes[i] - btes[s]
                vx = (delx*g)/(vt*(1.0-np.exp(-g*delt/vt)))
                vy = (dely*g  )/(vt*(1.0-np.exp(-g*delt/vt)))
                vzi = (((delz+vt*delt)*g)/(vt*(1.0-np.exp(-g*delt/vt)))) - vt
                vxm.append(vx)
                vym.append(vy)
                vzim.append(vzi)
            print("============ Replanned after", btes[3]-btes[2],
                  " timestamp=", btes[3])
        else:
            print("First")
            for i in range(s+1,e+1):
                delx = bpes[i][0] - bpes[s][0]
                dely = bpes[i][1] - bpes[s][1]
                delz = bpes[i][2] - bpes[s][2]
                delt = btes[i] - btes[s]
                vx = (delx*g)/(vt*(1.0-np.exp(-g*delt/vt)))
                vy = (dely*g  )/(vt*(1.0-np.exp(-g*delt/vt)))
                vzi = (((delz+vt*delt)*g)/(vt*(1.0-np.exp(-g*delt/vt)))) - vt
                vxm.append(vx)
                vym.append(vy)
                vzim.append(vzi)
        vx = np.mean(vxm)
        vy = np.mean(vym)
        vzi = np.mean(vzim)

        # Create message of projectile type
        projectile_msg = Projectile()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'odom_combined'
        projectile_msg.header = header
        projectile_msg.object_id = 0

        # Introduce errorz here (TODO: why)
        if len(bpes) <= 3 and publish_pmsg:
            errorz = 0.0
            print("Introducing Error of ", errorz)
            bpes[s][1] += errorz

        # Setting the variable in the message to computed results
        projectile_msg.position.x = bpes[s][0]
        projectile_msg.position.y = bpes[s][1]
        projectile_msg.position.z = bpes[s][2]
        projectile_msg.velocity.x = vx
        projectile_msg.velocity.y = vy
        projectile_msg.velocity.z = vzi

        # Setting local object's variable
        self.projectile_estimates.append([bpes[s][0],
                                         bpes[s][1],
                                         bpes[s][2],
                                         vx, vy, vzi])
        print("Compute Projectile:",
              [bpes[s][0], bpes[s][1], bpes[s][2], vx, vy, vzi])

        if publish_pmsg:
            pmsg.publish(projectile_msg)
            print(projectile_msg)
        # Time elapsed
        t2 = rospy.Time.now()
        print("Diff", (t2-t1).to_sec()*1000, t2)

        return projectile_msg

    

    # Helper function to extrapolate projectile to time
    def extrapolate_projectile_to_time(
            self,px,py,pz,
            vx,vy,vz,delt
            ):
        g = -self.gravity
        vt = self.vt
        ez = pz + (vt/g)*(vz+vt)*(1.0-np.exp(-(g*delt/vt)))-vt*delt
        ex = px + (vx*vt/g)*(1.0-np.exp(-g*delt/vt))
        ey = py + (vy*vt/g)*(1.0-np.exp(-g*delt/vt))
        return np.array([ex,ey,ez])

    # Helper function to compute accuracy
    def compute_accuracy(
            self,ipx,ipy,ipz,
            ivx,ivy,ivz,
            wall_x=1.0
            ):
        # Getting object's data
        ki_points = np.array(
                  self.ki_sess.k_ball_position_estimates).transpose()
        ki_times = np.array(
                 self.ki_sess.k_ball_time_estimates)\
                 -self.sc_sess.first_sc_timestamp
        # Compute velocity of GT projectile
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
            vzi = (delz-0.5*self.gravity*delt*delt)/delt
            k_vxm.append(vx)
            k_vym.append(vy)
            k_vzim.append(vzi)
        k_vx = np.mean(k_vxm)
        k_vy = np.mean(k_vym)
        k_vzi = np.mean(k_vzim)

        ki_delx = wall_x-ki_points[0,0]
        ki_delt = ki_delx/(k_vx)
        ki_point = self.extrapolate_projectile_to_time(ki_points[0,0],
                                                  ki_points[1,0],
                                                  ki_points[2,0],
                                                  k_vx,k_vy,k_vzi,
                                                  ki_delt)
        reference_points = np.concatenate(
                         [ki_points, ki_point.reshape(-1,1)], axis=1)
        extrapolated_points = np.array(
                            [self.extrapolate_projectile_to_time(ipx,ipy,ipz,
                                                            ivx,ivy,ivz,delt)
                            for delt in np.arange(ki_times[0]-0.030,
                                                  ki_delt+ki_times[0],
                                                  0.005)
                            ])
        extrapolated_points = extrapolated_points.transpose()

        if (extrapolated_points.shape[0]==0):
            print("Error Extrapolating Points")

        diff_list = []
        for i in range(reference_points.shape[1]):
            diff = np.min(
                 np.linalg.norm(extrapolated_points-reference_points[:,i:i+1],
                 axis=0))
            diff_list.append(diff)

        error = np.mean(diff_list)
        return error

###########################################################################
##############################WORKING MAIN#################################
    def listener( self ):
        # Init ROS node(process)
        rospy.init_node('c_p_calibrate')

        # Find TF
        self.sc_sess.sc_find_FP()
        # self.ki_sess.ki_find_FP()

        if METHOD_ID == 0:
            ## Using bounding box
            rospy.Subscriber("/zed_filtered/output", PointCloud2,
                            structure_callback,
                            callback_args=(self,self.sc_sess),
                            queue_size=10)
        elif METHOD_ID == 1:
            ## Using Color filtering
            rospy.Subscriber("/zed2i/filtered_point_cloud", PointCloud2,
                            structure_callback,
                            callback_args=(self,self.sc_sess),
                            queue_size=10)

        # Load pre-save params files
        # params = [0.404571, 0.13428446, 0.35198332,
        #           1.5798978885211274, -8.995515878771856,
        #           -0.34771949149796966, 0.01176908]
        # drag_coeff = 0.01176908 
        rospack = rospkg.RosPack()
        pkg_dir = rospack.get_path('shield_perception')
        param_filename = pkg_dir + "/data_dump/perception_params.npy"
        params = tuple(np.load(param_filename).tolist())
        print("Param file ", param_filename)
        print("Params loaded", params)

        # # Setup params
        # self.x_offset, self.y_offset, self.z_offset,\
        #     self.roll_offset, self.pitch_offset,\
        #     self.yaw_offset, self.drag_coeff = params
        # self.pitch_offset, self.roll_offset, self.yaw_offset =\
        #     self.pitch_offset*np.pi/180.0, self.roll_offset*np.pi/180.0,\
        #     self.yaw_offset*np.pi/180.0
        # self.res_pitch_A = np.array(
        #                  [[1,0,0,0],
        #                  [0,np.cos(self.pitch_offset),-np.sin(self.pitch_offset),0],
        #                  [0,np.sin(self.pitch_offset),np.cos(self.pitch_offset),0],
        #                  [0,0,0,1]])
        # self.res_yaw_A = np.array(
        #                  [[np.cos(self.yaw_offset),0,np.sin(self.yaw_offset),0],
        #                  [0,1,0,0],
        #                  [-np.sin(self.yaw_offset),0,np.cos(self.yaw_offset),0],
        #                  [0,0,0,1]])
        # self.res_roll_A = np.array(
        #                  [[np.cos(self.roll_offset),-np.sin(self.roll_offset),0,0],
        #                  [np.sin(self.roll_offset),np.cos(self.roll_offset),0,0],
        #                  [0,0,1,0],
        #                  [0,0,0,1]])
        rate = rospy.Rate(240) # 10hz

        # Start of the loop with loop control variable process_sc
        process_sc = True
        while not rospy.is_shutdown():
            # When projectile is computed
            if self.sc_sess.projectile_computed and process_sc:
                t1 = rospy.Time.now()
                self.sc_sess.sc_filter( self )
                t2 = rospy.Time.now()
                self.compute_projectile_2(
                    self.sc_sess.ball_position_estimates,
                    self.sc_sess.ball_time_estimates_filtered,
                    self.sc_sess.projectile_msg_pub,
                    self.sc_sess.projectile_marker_pub)
                process_sc = False
                print("Structure CORE TimeSTAMPS",
                      self.sc_sess.ball_time_estimates, "time diff",
                      (t2-t1).to_sec()*1000, t1.to_sec())
                
                # Visualization - Publish the points used to calculate probalo
                self.visualize_projectile_points(
                    self.sc_sess.ball_position_estimates,
                    self.sc_sess.projectile_marker_pub)
                self.visualize_collected_pc(
                    self.sc_sess.ball_pointcloud,
                    self.sc_sess.ball_pointcloud_rgb,
                    self.sc_sess.publisher)

                break

            # When kinect need replanning
            # if self.ki_sess.k_enable_replanning and\
            #         self.ki_sess.k_kinect_replanning_available:
            #     self.sc_sess.sc_filter( self )
            #     self.compute_projectile_2(
            #         self.sc_sess.ball_position_estimates
            #         +np.array(self.ki_sess.k_ball_position_estimates).tolist(),
            #         self.sc_sess.ball_time_estimates_filtered
            #         +self.ki_sess.k_ball_time_estimates,
            #         self.sc_sess.projectile_msg_pub,
            #         self.sc_sess.projectile_marker_pub)
            #     k_enable_replanning = False
            # if self.ki_sess.k_projectile_computed:
            #     break
            rate.sleep()
##############################DONE WORKING#################################
###########################################################################

class SCSession ( object ):
    '''
    Session for structure core to capture the projectile
    '''
    def __init__ ( self ):
        # ROS component
        self.publisher = rospy.Publisher('/sc/rgbd/filtered_points',
                                        PointCloud2,
                                        queue_size = 1)
        self.projectile_msg_pub = rospy.Publisher("projectile", 
                                                  Projectile,
                                                  queue_size=1)
        self.projectile_marker_pub = rospy.Publisher("/projectile_vis",
                                                     MarkerArray,
                                                     queue_size=3)
        # Data
        self.A_FP_SC = None
        self.first_sc_timestamp = None
        self.ball_position_estimates_base = []
        self.ball_position_estimates = []
        self.ball_time_estimates = []
        self.ball_pointcloud = []
        self.ball_pointcloud_rgb = []
        self.order = []     # Indices of least errors poses, will be a np array
        # Control
        self.projectile_computed = False

    # Function to find FP of sc
    def sc_find_FP( self ):
        # Old setup: hardcoded trans and rot, fix the error via calibration
        # # Const trans and rot for structure core
        # tf_trans = [6.01944448, -0.52128749,  0.55443587]
        # tf_rot = [0.4995405,  0.45499593, 0.51035028, 0.53195919]
        # trans_mat = tf.transformations.translation_matrix(tf_trans)
        # rot_mat = tf.transformations.quaternion_matrix(tf_rot)
        # self.A_FP_SC = np.dot(trans_mat, rot_mat)
        # print("Transformation(SC) Loaded", self.A_FP_SC)

        tf_found = False
        rate = rospy.Rate(10.0)
        tf_trans = None
        tf_rot = None
        tf_listener = tf.TransformListener()
        # Find tf (transform) listener
        while not tf_found:
            try:
                if METHOD_ID == 0:
                    # Transform is no longer needed because it's been done
                    # in passthrough filering
                    (tf_trans,tf_rot) = tf_listener.lookupTransform(
                                        "/base_link",
                                        "/base_link",
                                        rospy.Time(0))
                elif METHOD_ID == 1:
                    # Look for trans and rot for zed2i_camera_center
                    # from base_link
                    (tf_trans,tf_rot) = tf_listener.lookupTransform(
                                        "/base_link",
                                        "/zed2i_left_camera_frame",
                                        rospy.Time(0))
                print("translation ", tf_trans)
                print("rotation ", tf_rot)
                tf_found = True
            except Exception as e:
                print("TF not found", e)
            rate.sleep()
        print("Found TF")

        trans_mat = tf.transformations.translation_matrix(tf_trans)
        rot_mat = tf.transformations.quaternion_matrix(tf_rot)
        self.A_FP_SC = np.dot(trans_mat, rot_mat)

    # Function to process filtering the points collected
    def sc_filter( self, track_obj ):
        self.ball_position_estimates_base_filtered = []
        self.ball_time_estimates_filtered = []
        self.errors = []
        # Calculate Errors
        for i in range(NUM_FRAME):
            error = np.mean(
                  np.linalg.norm(
                  self.ball_pointcloud[i]
                  -self.ball_position_estimates_base[i].reshape((1,-1)),
                  axis=1))
            self.errors.append(error)
        # Picked three with least error
        self.order = np.argsort(self.errors)
        print("======== Order======== ", self.order)
        for i in range(NUM_FRAME):
            if i in self.order[0:3]:
                self.ball_position_estimates_base_filtered.append(
                    self.ball_position_estimates_base[i])
                self.ball_time_estimates_filtered.append(
                    self.ball_time_estimates[i])

        ## This step is no longer needed. The transform is done during callback
        # sc_points = np.concatenate(
        #           [np.array(self.ball_position_estimates_base_filtered).transpose(),
        #           np.ones((1,len(self.ball_position_estimates_base_filtered)))],
        #           axis=0)
        # base_frame_points = np.linalg.multi_dot(
        #                  [self.A_FP_SC, track_obj.res_roll_A,
        #                  track_obj.res_pitch_A, track_obj.res_yaw_A,
        #                  sc_points])
        # base_frame_points += np.array(
        #                  [[track_obj.x_offset],
        #                  [track_obj.y_offset],
        #                  [track_obj.z_offset],
        #                  [0.0]])
        self.ball_position_estimates = self.ball_position_estimates_base_filtered
        self.first_sc_timestamp = self.ball_time_estimates_filtered[0]


if __name__ == '__main__':
    track_ball = TrackBall()
    track_ball.listener()
