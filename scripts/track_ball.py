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
from visualization_msgs.msg import MarkerArray
import tf

from dynamic_reconfigure.server import Server
# from shield_perception.cfg import calibrateConfig


##########################################################################
#############################Function#####################################
# Helper function to turn xyzrgb array to pc2 data structure
def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)
    buf = []
    # Process args
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
    # Setting up message's fields and opitions
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

# Helper function for ros to point cloud list
def ros_to_pcl(ros_cloud):
    points_list = []
    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2]])
    return np.array(points_list)

# Kinect callback function
def kinect_callback( ros_cloud, args ):

    track_obj = args[0]
    sc_obj = args[1]
    ki_obj = args[2]

    # Return directly if sc have not computed yet or ki is done
    if ki_obj.k_projectile_computed or not sc_obj.projectile_computed:
        return

    # Translate ros's data to pointcloud
    ball_points = ros_to_pcl( ros_cloud )
    # Filter out some input
    if len( ki_obj.k_ball_position_estimates ) == 0 and\
            ball_points.shape[0] < 1:
        return
    elif ball_points.shape[0] < 50:
        return

    # Process data collected
    ball_position = np.mean(ball_points, axis=0)
    hom_ball_position = np.ones((4,1))
    hom_ball_position[0,0] = ball_position[0]
    hom_ball_position[2,0] = ball_position[2]
    hom_ball_position[1,0] = ball_position[1]
    # Caculate ball position with repect to PC2
    P_FP_B = np.dot( ki_obj.A_FP_KI, hom_ball_position )
    ball_position_in_PR2 = P_FP_B[0:3,0]/P_FP_B[3,0]
    print("kinect Ball points",
          ball_points.shape[0],
          ball_position_in_PR2,
          ros_cloud.header.stamp.to_sec())

    # Update data structure only if structure core is done
    if sc_obj.projectile_computed: 
        ki_obj.k_ball_time_estimates.append(ros_cloud.header.stamp.to_sec())
        ki_obj.k_ball_position_estimates.append(ball_position_in_PR2)
    ki_obj.k_ball_pointcloud.append(ball_points)

    # Enable replanning when there's no enough data points
    if len(ki_obj.k_ball_position_estimates) >= 1 and not\
            ki_obj.k_projectile_computed:
        ki_obj.k_kinect_replanning_available = True

    # Save data to pickle files when enough data is obtained
    if len(ki_obj.k_ball_position_estimates) >= 3 and not\
            ki_obj.k_projectile_computed:
        print("==============kinect==========================")
        if sc_obj.projectile_computed:
            save_dict = {}
            save_dict['sc'] = ( sc_obj.ball_position_estimates_base,
                                sc_obj.ball_time_estimates )
            save_dict['sc_points'] = sc_obj.ball_pointcloud
            save_dict['ki'] = ( ki_obj.k_ball_position_estimates,
                                ki_obj.k_ball_time_estimates )
            save_dict['ki_points'] = ki_obj.k_ball_pointcloud
            save_dict['TF'] = ki_obj.A_FP_KI
            rospack = rospkg.RosPack()
            pkg_dir = rospack.get_path('shield_perception')
            filename = pkg_dir + "/data_dump/calib_data_runs/" +\
                       time.strftime("%Y%m%d-%H%M%S") + '.pickle'

            # Create pickle file in specified path
            with open(filename, 'wb') as handle:
                pickle.dump(save_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
                print("Saving raw files to ", filename)
            for estimates in track_obj.projectile_estimates:
                # print(estimates, "Estimates")
                print(track_obj.compute_accuracy(
                      estimates[0],estimates[1],
                      estimates[2],estimates[3],
                      estimates[4],estimates[5])*100.0,
                      "Errors (cm)")
                print("Time Diff (s) = ",
                      ki_obj.k_ball_time_estimates[0]-\
                      sc_obj.ball_time_estimates[-1])
        ki_obj.k_projectile_computed = True

    # if G_DEBUG: debugging info
    dtype_list =\
        rnp.point_cloud2.fields_to_dtype(ros_cloud.fields,
                                         ros_cloud.point_step)
    filtered_msg =\
        xyzrgb_array_to_pointcloud2(ball_points,
                                    np.zeros(ball_points.shape,
                                             dtype=np.float32),
                                    ros_cloud.header.stamp,
                                    ros_cloud.header.frame_id,
                                    ros_cloud.header.seq)
    # Pulish
    ki_obj.k_publisher.publish(filtered_msg)

# Call back function for structure core
def structure_callback( ros_cloud, args ):
    track_obj = args[0]
    sc_obj = args[1]
    ki_obj = args[2]
    # Getting spatial limit from track ball object
    y_pos_lim = track_obj.y_pos_lim
    y_neg_lim = track_obj.y_neg_lim
    z_pos_lim = track_obj.z_pos_lim
    z_neg_lim = track_obj.z_neg_lim
    # If projectile already computed, just return
    if sc_obj.projectile_computed:
        return
    # Converting point cloud to python vectors
    point_list = []
    pc = rnp.numpify(ros_cloud)
    points = np.zeros((pc.shape[0],3))
    points[:,0] = pc['x']
    points[:,1] = pc['y']
    points[:,2] = pc['z']
    # Filter out some points that exceed spatial limit
    ball_points = points[points[:,1]>y_neg_lim,:]
    ball_points = ball_points[ball_points[:,1]<y_pos_lim,:]
    ball_points = ball_points[ball_points[:,2]>z_neg_lim,:]
    ball_points = ball_points[ball_points[:,2]<z_pos_lim,:]
    # Only count the points when sc capture enough (15+) points
    if(ball_points.shape[0]<15):
        return
    ball_position = np.mean(ball_points, axis=0)
    # Save data into data structure in object
    sc_obj.ball_time_estimates.append(ros_cloud.header.stamp.to_sec())
    sc_obj.ball_position_estimates_base.append(ball_position)
    sc_obj.ball_pointcloud.append(ball_points)
    # Print out data for debugging purpose
    print("Structure Core Ball points",
          ball_points.shape[0], ball_position,
          ros_cloud.header.stamp.to_sec())
    # If collected more than 5 points, mark projectile as computed
    # for structure core
    if len(sc_obj.ball_position_estimates_base) >= 5 and not\
            sc_obj.projectile_computed:
        print("=============STRUCTURE CORE===========================",
              rospy.Time.now().to_sec())
        sc_obj.projectile_computed = True
    # If debug mode is set, publish data collected to filterd_msg
    if track_obj.g_debug:
        dtype_list = rnp.point_cloud2.fields_to_dtype(
                   ros_cloud.fields,
                   ros_cloud.point_step )
        filtered_msg = xyzrgb_array_to_pointcloud2(
                     ball_points,
                     np.zeros(ball_points.shape, dtype=np.float32),
                     ros_cloud.header.stamp,
                     "launcher_camera",
                     ros_cloud.header.seq )
        sc_obj.publisher.publish(filtered_msg)
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
        self.g_debug = True

        # Spatial
        self.x_cen = -0.4515569
        self.x_width = 0.768414*0.8
        self.y_cen = -0.0709
        self.y_width = 1.2218*0.97
        self.z_pos_lim = 3.
        self.z_neg_lim = 0.01
        self.vel_scale = 1.0 # Not used 
        self.z_offset = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.roll_offset = 0.0
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.x_pos_lim = self.x_cen + self.x_width/2.0
        self.x_neg_lim = self.x_cen - self.x_width/2.0
        self.y_pos_lim = self.y_cen + self.y_width/2.0
        self.y_neg_lim = self.y_cen - self.y_width/2.0
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
        self.ki_sess = KISession()
        self.sc_sess = SCSession()

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
            for i in [e]:
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
        self.ki_sess.ki_find_FP()

        # Subscribe to cam output with callback functions
        rospy.Subscriber("/kinect_filtered/output", PointCloud2,
                         kinect_callback,
                         callback_args=(self,self.sc_sess,self.ki_sess))
        rospy.Subscriber("/external_filtered/output", PointCloud2,
                         structure_callback,
                         callback_args=(self,self.sc_sess,self.ki_sess))

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

        # Setup params
        self.x_offset, self.y_offset, self.z_offset,\
            self.roll_offset, self.pitch_offset,\
            self.yaw_offset, self.drag_coeff = params
        self.pitch_offset, self.roll_offset, self.yaw_offset =\
            self.pitch_offset*np.pi/180.0, self.roll_offset*np.pi/180.0,\
            self.yaw_offset*np.pi/180.0
        self.res_pitch_A = np.array(
                         [[1,0,0,0],
                         [0,np.cos(self.pitch_offset),-np.sin(self.pitch_offset),0],
                         [0,np.sin(self.pitch_offset),np.cos(self.pitch_offset),0],
                         [0,0,0,1]])
        self.res_yaw_A = np.array(
                         [[np.cos(self.yaw_offset),0,np.sin(self.yaw_offset),0],
                         [0,1,0,0],
                         [-np.sin(self.yaw_offset),0,np.cos(self.yaw_offset),0],
                         [0,0,0,1]])
        self.res_roll_A = np.array(
                         [[np.cos(self.roll_offset),-np.sin(self.roll_offset),0,0],
                         [np.sin(self.roll_offset),np.cos(self.roll_offset),0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
        rate = rospy.Rate(600) # 10hz

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

            # When kinect need replanning
            if self.ki_sess.k_enable_replanning and\
                    self.ki_sess.k_kinect_replanning_available:
                self.sc_sess.sc_filter( self )
                self.compute_projectile_2(
                    self.sc_sess.ball_position_estimates
                    +np.array(self.ki_sess.k_ball_position_estimates).tolist(),
                    self.sc_sess.ball_time_estimates_filtered
                    +self.ki_sess.k_ball_time_estimates,
                    self.sc_sess.projectile_msg_pub,
                    self.sc_sess.projectile_marker_pub)
                k_enable_replanning = False
            if self.ki_sess.k_projectile_computed:
                break
            rate.sleep()
##############################DONE WORKING#################################
###########################################################################

class KISession ( object ):
    '''
    Session for kinect cam to capture the projectile
    '''
    def __init__ ( self ):
        # ROS component
        self.k_publisher = rospy.Publisher('/kinect/filtered_points',
                                          PointCloud2,
                                          queue_size = 1)
        self.k_projectile_marker_pub = rospy.Publisher("kinect_projectile_vis",
                                                       MarkerArray,
                                                       queue_size=10)
        # Data
        self.A_FP_KI = None
        self.k_ball_position_estimates = []
        self.k_ball_time_estimates = []
        self.k_ball_pointcloud = []
        # Control
        self.k_projectile_computed = False
        self.k_enable_replanning = True
        self.k_kinect_replanning_available = False

    def ki_find_FP( self ):
        tf_found = False
        rate = rospy.Rate(10.0)
        k_tf_trans = None
        k_tf_rot = None
        tf_listener = tf.TransformListener()
        # Find tf (transform) listener
        while not tf_found:
            try:
                # Look for trans and rot for camera_depth_optical_frame
                # from base_footprint
                (k_tf_trans,k_tf_rot) = tf_listener.lookupTransform(
                                      "base_footprint",
                                      "/camera_depth_optical_frame",
                                      rospy.Time(0))
                print("translation ", k_tf_trans)
                print("rotation ", k_tf_rot)
                tf_found = True
            except Exception as e:
                print("TF not found", e)
            rate.sleep()
        print("Found TF")

        # Previously found tf examples
        # k_tf_trans = [-0.15783123802738946, 0.01273613573344717,
        #               1.4690869133055597]
        # k_tf_rot = [-0.4749945606732899, 0.5031187344211056,
        #             -0.5249774084103889, 0.4956313418903257]

        # Multiplying trans and rot to get relative to base_footprint
        trans_mat = tf.transformations.translation_matrix(k_tf_trans)
        rot_mat = tf.transformations.quaternion_matrix(k_tf_rot)
        self.A_FP_KI = np.dot(trans_mat, rot_mat)


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
        self.projectile_marker_pub = rospy.Publisher("projectile_vis",
                                                     MarkerArray,
                                                     queue_size=10)
        # Data
        self.A_FP_SC = None
        self.first_sc_timestamp = None
        self.ball_position_estimates_base = []
        self.ball_position_estimates = []
        self.ball_time_estimates = []
        self.ball_pointcloud = []
        # Control
        self.projectile_computed = False

    # Function to find FP of sc
    def sc_find_FP( self ):
        # Const trans and rot for structure core
        tf_trans = [6.01944448, -0.52128749,  0.55443587]
        tf_rot = [0.4995405,  0.45499593, 0.51035028, 0.53195919]
        trans_mat = tf.transformations.translation_matrix(tf_trans)
        rot_mat = tf.transformations.quaternion_matrix(tf_rot)
        self.A_FP_SC = np.dot(trans_mat, rot_mat)
        print("Transformation(SC) Loaded", self.A_FP_SC)

    # Function to process filtering the points collected
    def sc_filter( self, track_obj ):
        self.ball_position_estimates_base_filtered = []
        self.ball_time_estimates_filtered = []
        self.errors = []
        # Calculate Errors
        for i in range(5):
            error = np.max(
                  np.linalg.norm(
                  np.array(self.ball_pointcloud[i])
                  -self.ball_position_estimates_base[i].reshape((1,-1)),
                  axis=1))
            self.errors.append(error)
        # Picked three with least error
        order = np.argsort(self.errors)
        print("======== Order======== ", order)
        for i in range(5):
            if i in order[0:3]:
                self.ball_position_estimates_base_filtered.append(
                    self.ball_position_estimates_base[i])
                self.ball_time_estimates_filtered.append(
                    self.ball_time_estimates[i])

        sc_points = np.concatenate(
                  [np.array(self.ball_position_estimates_base_filtered).transpose(),
                  np.ones((1,len(self.ball_position_estimates_base_filtered)))],
                  axis=0)
        pr2_frame_points = np.linalg.multi_dot(
                         [self.A_FP_SC, track_obj.res_roll_A,
                         track_obj.res_pitch_A, track_obj.res_yaw_A,
                         sc_points])
        pr2_frame_points += np.array(
                         [[track_obj.x_offset],
                         [track_obj.y_offset],
                         [track_obj.z_offset],
                         [0.0]])
        self.ball_position_estimates = pr2_frame_points[0:3,:].transpose().tolist()
        self.first_sc_timestamp = self.ball_time_estimates_filtered[0]


if __name__ == '__main__':
    track_ball = TrackBall()
    track_ball.listener()
