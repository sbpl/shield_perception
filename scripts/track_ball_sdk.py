#!/usr/bin/env python
import cv2
import numpy as np
# import open3d as o3d
import os
import pyzed.sl as sl
import rospy
from scipy.optimize import least_squares
from sensor_msgs.msg import PointCloud2, PointField
from shield_planner_msgs.msg import Projectile
import std_msgs.msg
import sys
from visualization_msgs.msg import MarkerArray, Marker

import pdb

# Relative Imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"camera_calibration"))
from helpers import *
from constants import *

# MACRO
OUTLIER_REJECT=1
BOUNDING_FILTER=1
DIST_THRESHOLD=4.5
MIN_PIXEL=15
PUBLISH_PROJ=1
METHOD_ID=1         #0 = native bounding box (aborted), 1 = color detection, 2 = open3d bounding box (aborted)
DEBUG=0
VISUAL=0
SAVE_IMG=0
RES=1

# min_radius = 1  # Minimum radius of the ball
# max_radius = 30  # Maximum radius of the ball

# Global Vars
Num_Frame = 5
measurements = []
stamps = []
finish_stamp = 0
pc_xyz_list = []
pc_rgb_list = []


##########################################################################
#############################Function#####################################
## Visualization
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

# Function to publish points used to estimate the projectile
def visualize_projectile_points( bpes, pmmsg ):
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
        marker.pose.position.x = es[1]
        marker.pose.position.y = es[2] 
        marker.pose.position.z = es[3]
        markers.markers.append(marker)
    pmmsg.publish(markers)

# Function to visualized pc2 that are collected
def visualize_collected_pc( pc_list, pc_rgb_list, pub ):
    pc_all = np.zeros((0,3))
    pc_rgb_all = np.zeros((0,1))
    for i in range(len(pc_list)):
        pc_all = np.vstack((pc_all, np.array(pc_list[i])))            

        # Trying to convert BGR to RGB
        # pc_rgb = np.array(pc_rgb_list[i]).reshape(-1, 3)[:, [2, 1, 0]]
        # pc_rgb_all = np.vstack((pc_rgb_all, pc_rgb.reshape(-1, 3)))
        
        # pdb.set_trace()
        pc_rgb_all = np.vstack((pc_rgb_all, np.array(pc_rgb_list[i]).reshape(-1,1)))

    # Publishing
    filtered_msg = xyzrgb_array_to_pointcloud2(
                pc_all,
                pc_rgb_all,
                rospy.Time.now(),
                "odom_combined")
    pub.publish(filtered_msg)

##########################################################################
## Projectile Estimation
# Define the mathematical model for the trajectory
def trajectory_model(params, t):
    x0, y0, z0, vx0, vy0, vz0 = params
    t_squared = t**2
    x = x0 + vx0 * t
    y = y0 + vy0 * t
    z = z0 + vz0 * t - 0.5 * 9.81 * t_squared
    return np.array([x, y, z])

# Define the error function to minimize (sum of squared errors)
def error_function(params):
    errors = []
    for t, observed_x, observed_y, observed_z in measurements:
        predicted_position = trajectory_model(params, t)
        errors.extend([observed_x - predicted_position[0], observed_y - predicted_position[1], observed_z - predicted_position[2]])
    return errors

# Function to estimate the trajectory parameters using least squares
def estimate_trajectory(measurements):
    # Initial guess for the model parameters (adjust as needed)
    initial_params = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Use least squares optimization to estimate the model parameters
    result = least_squares(error_function, initial_params)

    # Extract the optimized parameters representing initial position and velocity
    x0, y0, z0, vx0, vy0, vz0 = result.x
    return (x0, y0, z0, vx0, vy0, vz0)

##########################################################################
##########################################################################
## Working Main

def main():
    # Setup ROS publishers
    if DEBUG:
        rospy.init_node('track_ball_sdk', log_level=rospy.DEBUG)
    else:
        rospy.init_node('track_ball_sdk')
    pc_pub = rospy.Publisher('/sc/rgbd/filtered_points',
                            PointCloud2,
                            queue_size = 1)
    projectile_msg_pub = rospy.Publisher("projectile", 
                                        Projectile,
                                        queue_size=1)
    projectile_marker_pub = rospy.Publisher("/projectile_vis",
                                            MarkerArray,
                                            queue_size=Num_Frame)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    if RES == 0:
        init.camera_resolution = sl.RESOLUTION.VGA
        init.camera_fps=100
    elif RES == 1:
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps=60
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    # init.depth_stabilization=50
    init.coordinate_units = sl.UNIT.METER # Use meter units (for depth measurements)
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.texture_confidence_threshold = 100

    # Capture images
    i = 0
    image= sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    confidence_map = sl.Mat()

    count = 0

    if SAVE_IMG:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"camera_calibration/color_data/")

    while i < 500:
        #A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Time Stamps of getting the Frame
            stamp_temp = rospy.Time.now().to_sec()
            i = i + 1
            fps = zed.get_current_fps()
            # rospy.loginfo("Frame Rate: {} FPS".format(fps))
            print("FRAMERATE: {} FPS".format(fps))

            # continue

            # Retrieving Data
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

            # Turning Data into np array
            pc_xyz_np = point_cloud.get_data()[:,:,:3]
            pc_rgb_np = point_cloud.get_data()[:,:,3]
            confidence_np = confidence_map.get_data()

            # continue
            ## stamp 1: after retrieving data
            # stamp_1 = rospy.Time.now().to_sec()

            if DEBUG:
                rospy.logdebug("TIME SPENT IN RETRIEVING: {}".format(stamp_1-stamp_temp))

            
            # Detection Method:
            '''
            Input: point cloud as np array (ROW x COL x 3), Left cam image
            Output: poi of pointclouds and confidence
            '''
            # Setting up spatial limit
            # Define the boundaries of the virtual bounding box
            bbox_x_min = 0.8  
            bbox_x_max = DIST_THRESHOLD 
            bbox_y_min = -2.0 
            bbox_y_max = 2.0  
            bbox_z_min = 0.0
            bbox_z_max = 2.8
            # x_neg_lim = 1.0
            # x_pos_lim = 4.5
            # y_neg_lim = -2.0
            # y_pos_lim = 2.0
            # z_neg_lim = 0.0
            # z_pos_lim = 2.8
            if METHOD_ID == 0:
                continue
                # Reshape
                # confidence_poi_all = confidence_np.reshape((-1,1))
                # pc_xyz_poi_all = pc_xyz_np.reshape((-1,3))
                # pc_rgb_poi_all = pc_rgb_np.reshape((-1,1))
                # valid_ind = np.logical_not(np.isnan(pc_xyz_np[:,:,0]))

                # pdb.set_trace()

                # Throw out NAN
                # confidence_poi = confidence_np[valid_ind]
                # pc_xyz_poi = pc_xyz_np[valid_ind,:]
                # pc_rgb_poi = pc_rgb_np[valid_ind]
                # confidence_poi = confidence_poi_all[valid_ind,:]
                # pc_xyz_poi = pc_xyz_poi_all[valid_ind,:]
                # pc_rgb_poi = pc_rgb_poi_all[valid_ind,:]

                # TF
                # pc_xyz_poi = np.dot(T_BASE_TO_LEFT, np.append(pc_xyz_poi, np.ones((pc_xyz_poi.shape[0],1)), axis=1).transpose())[0:3,:].transpose()

                # # Passthrough filter
                # ind_x = np.intersect1d(np.where(pc_xyz_poi[:,0] > bbox_x_min)[0], np.where(pc_xyz_poi[:,0] < bbox_x_max)[0])
                # ind_y = np.intersect1d(np.where(pc_xyz_poi[:,1] > bbox_y_min)[0], np.where(pc_xyz_poi[:,1] < bbox_y_max)[0])
                # ind_z = np.intersect1d(np.where(pc_xyz_poi[:,2] > bbox_z_min)[0], np.where(pc_xyz_poi[:,2] < bbox_z_max)[0])
                # filtered_ind = np.intersect1d(np.intersect1d(ind_x, ind_y), ind_z)

                # # Output
                # confidence_poi = confidence_poi[filtered_ind,:]
                # pc_xyz_poi = pc_xyz_poi[filtered_ind,:]
                # pc_rgb_poi = pc_rgb_poi[filtered_ind,:]

            elif METHOD_ID == 1:
                # Convert ZED Mat objects to numpy arrays
                image_ocv = image.get_data()
                hsv_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2HSV)

                # Define the adjusted range for bright orange color in HSV
                lower_bound = np.array([5, 75, 100])
                upper_bound = np.array([30, 255, 255])

                # Create a binary mask for orange color in HSV
                mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
                where = np.where(mask == 255)

                # DEBUG: Visualize masked image
                if VISUAL:
                    image_masked = cv2.bitwise_and(image_ocv, image_ocv, mask=mask)
                    cv2.namedWindow('image1')
                    cv2.imshow("image1", image_masked)
                    cv2.waitKey(1)

                # Masked PC and Confidence
                confidence_poi_all = confidence_np[where[0], where[1]].reshape((-1,1))
                pc_xyz_poi_all = pc_xyz_np[where[0], where[1], :].reshape((-1,3))
                pc_rgb_poi_all = pc_rgb_np[where[0], where[1]].reshape((-1,1))
                valid_ind = np.logical_not(np.isnan(pc_xyz_poi_all[:,0]))

                # Throw out NAN
                confidence_poi = confidence_poi_all[valid_ind,:]
                pc_xyz_poi = pc_xyz_poi_all[valid_ind,:]
                pc_rgb_poi = pc_rgb_poi_all[valid_ind,:]
                # TF
                pc_xyz_poi = np.dot(T_BASE_TO_LEFT, np.append(pc_xyz_poi, np.ones((pc_xyz_poi.shape[0],1)), axis=1).transpose())[0:3,:].transpose()

            elif METHOD_ID == 2:
                continue
                # # Reshape
                # confidence_poi_all = confidence_np.reshape((-1,1))
                # pc_xyz_poi_all = pc_xyz_np.reshape((-1,3))
                # pc_rgb_poi_all = pc_rgb_np.reshape((-1,1))

                # # Open3d pointcloud processing
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pc_xyz_poi_all)
                # bounds = [[bbox_x_min, bbox_x_max], [bbox_y_min, bbox_y_max], [bbox_z_min, bbox_z_max]]
                
                # # TODO: Aborting. This is taking too long

                # # Reshape
                # # confidence_poi_all = confidence_np
                # # pc_xyz_poi_all = pc_xyz_np
                # # pc_rgb_poi_all = pc_rgb_np
                # # valid_ind = np.logical_not(np.isnan(pc_xyz_poi_all[:,0]))

                # # Throw out NAN
                # # confidence_poi = confidence_poi_all[valid_ind,:]
                # # pc_xyz_poi = pc_xyz_poi_all[valid_ind,:]
                # # pc_rgb_poi = pc_rgb_poi_all[valid_ind,:]

            
            ## stamp 2: after detection method
            # stamp_2 = rospy.Time.now().to_sec()

            if DEBUG:
                rospy.logdebug("TIME SPENT IN DETECTION:  {}".format(stamp_2-stamp_1))

            # Bounding box filtering
            if BOUNDING_FILTER:
                # Throw out the pc that's not in the spatial limit
                ind_x = np.intersect1d(np.where(pc_xyz_poi[:,0] > bbox_x_min)[0], np.where(pc_xyz_poi[:,0] < bbox_x_max)[0])
                ind_y = np.intersect1d(np.where(pc_xyz_poi[:,1] > bbox_y_min)[0], np.where(pc_xyz_poi[:,1] < bbox_y_max)[0])
                ind_z = np.intersect1d(np.where(pc_xyz_poi[:,2] > bbox_z_min)[0], np.where(pc_xyz_poi[:,2] < bbox_z_max)[0])
                filtered_ind = np.intersect1d(np.intersect1d(ind_x, ind_y), ind_z)

                pc_xyz_poi = pc_xyz_poi[filtered_ind,:]
                pc_rgb_poi = pc_rgb_poi[filtered_ind,:]
                confidence_poi = confidence_poi[filtered_ind,:]


            # Outlier rejection
            if OUTLIER_REJECT:
                x_mean = np.mean(pc_xyz_poi[:,0])
                y_mean = np.mean(pc_xyz_poi[:,1])
                z_mean = np.mean(pc_xyz_poi[:,2])
                x_std = np.std(pc_xyz_poi[:,0])
                y_std = np.std(pc_xyz_poi[:,1])
                z_std = np.std(pc_xyz_poi[:,2])
                ind_x = np.intersect1d(np.where(pc_xyz_poi[:,0] > x_mean - 2*x_std)[0], np.where(pc_xyz_poi[:,0] < x_mean + 2*x_std)[0])
                ind_y = np.intersect1d(np.where(pc_xyz_poi[:,1] > y_mean - 2*y_std)[0], np.where(pc_xyz_poi[:,1] < y_mean + 2*y_std)[0])
                ind_z = np.intersect1d(np.where(pc_xyz_poi[:,2] > z_mean - 2*z_std)[0], np.where(pc_xyz_poi[:,2] < z_mean + 2*z_std)[0])
                ind = np.intersect1d(np.intersect1d(ind_x, ind_y), ind_z)
                pc_xyz_poi = pc_xyz_poi[ind,:]
                pc_rgb_poi = pc_rgb_poi[ind,:]
                confidence_poi = confidence_poi[ind,:]

            ## stamp 3: after detection method
            # stamp_3 = rospy.Time.now().to_sec()

            if DEBUG:
                rospy.logdebug("TIME SPENT IN FILTERING:  {}".format(stamp_3-stamp_2))

            # Only continue if there are more than MIN_PIXEL points (30+)
            if pc_xyz_poi.shape[0] < MIN_PIXEL:
                continue

            # Calculate the mean of the depth
            mean_X = np.mean(pc_xyz_poi[:,0])
            mean_Y = np.mean(pc_xyz_poi[:,1])
            mean_Z = np.mean(pc_xyz_poi[:,2])
            mean_Conf = np.mean(confidence_poi[:])
            if DEBUG:
                print("Mean Depth: {}".format(mean_X))

            # Check if the point is inside the bounding box
            # if bbox_x_min <= mean_X <= bbox_x_max and \
            #    bbox_y_min <= mean_Y <= bbox_y_max and \
            #    bbox_z_min <= mean_Z <= bbox_z_max:
                # Point is inside the bounding box, process it
            
            if mean_X != 0 and mean_X < DIST_THRESHOLD:
            # if mean_X != 0 and mean_X < DIST_THRESHOLD and mean_Conf < 50:
                if SAVE_IMG:
                    image_masked = cv2.bitwise_and(image_ocv, image_ocv, mask=mask)
                    img_mk_path = img_path + "img_mk_" + str(count) + ".jpg"
                    img_og_path = img_path + "img_og_" + str(count) + ".jpg"
                    cv2.imwrite(img_mk_path, image_masked)
                    cv2.imwrite(img_og_path, image_ocv)

                # Record Time stamps
                count = count + 1
                if count == 1:
                    t = 0.0
                    stamps.append(stamp_temp)
                else:
                    t = stamp_temp-stamps[0]
                    stamps.append(stamp_temp)

                # Storing Data
                print("t: {:.5f}, X: {:.5f}, Y: {:.5f}, Z: {:.5f}, Conf: {:.3f}".format(t, mean_X, mean_Y, mean_Z, mean_Conf))
                measurements.append((t, mean_X, mean_Y, mean_Z))
                pc_xyz_list.append(pc_xyz_poi)
                pc_rgb_list.append(pc_rgb_poi)

            # Collected Enough Points
            if count >= Num_Frame:
                # Perform trajectory estimation here using the measurements
                estimated_params = estimate_trajectory(measurements)
                print(f"Estimated Parameters (XYZ,VxVyVz): \n{estimated_params}")

                # Publish Projectile Msg            
                projectile_msg = Projectile()
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'odom_combined'
                projectile_msg.header = header
                projectile_msg.object_id = 0

                t_perc = rospy.Time.now().to_sec() - stamps[0]
                t_squared = t_perc**2
                x = estimated_params[0] + estimated_params[3]*t_perc
                y = estimated_params[1] + estimated_params[4]*t_perc
                z = estimated_params[2] + estimated_params[5]*t_perc - 0.5 * 9.81 * t_squared
                vz = estimated_params[5] - 9.81 * t_perc
                # x = x0 + vx0 * t
                # y = y0 + vy0 * t
                # z = z0 + vz0 * t - 0.5 * 9.81 * t_squared
                # Setting the variable in the message to computed results
                projectile_msg.position.x = x
                projectile_msg.position.y = y
                projectile_msg.position.z = z
                projectile_msg.velocity.x = estimated_params[3]
                projectile_msg.velocity.y = estimated_params[4]
                projectile_msg.velocity.z = vz

                # projectile_msg.position.x = estimated_params[0]
                # projectile_msg.position.y = estimated_params[1]
                # projectile_msg.position.z = estimated_params[2]
                # projectile_msg.velocity.x = estimated_params[3]
                # projectile_msg.velocity.y = estimated_params[4]
                # projectile_msg.velocity.z = estimated_params[5]


                if PUBLISH_PROJ:
                    projectile_msg_pub.publish(projectile_msg)
                    finish_stamp = rospy.Time.now().to_sec()
                    rospy.logwarn("Publishing projectile!")
                    print(projectile_msg)

                break

            # Code to pick color on-click on image
            # def click_event(event, x, y,  flags, params):
            #     if event == cv2.EVENT_LBUTTONDBLCLK:
            #         print('Col: ', x, ' Row: ', y)
            #         print(hsv_image[y, x])
            
            # cv2.namedWindow('image')
            # cv2.setMouseCallback('image',click_event)
            # cv2.imshow("image", image_ocv)
            # cv2.waitKey(1)
            

    # Cleanup ZED and CV
    cv2.destroyAllWindows()
    zed.close()

    # Visualization Publishing and Time profiling
    if count == Num_Frame:
        visualize_projectile_points(measurements, projectile_marker_pub)
        visualize_collected_pc(pc_xyz_list, pc_rgb_list, pc_pub)

        # Summary of the run
        print("*******************************************************")
        for ind, frame in enumerate(measurements):
            print("Point {}: Number of Pixels = {}".format(ind, pc_xyz_list[ind].shape[0]))
            print("Time Stamp: {}".format(stamps[ind]))
            print("t: {:.5f}, X: {:.5f}, Y: {:.5f}, Z: {:.5f}".format(frame[0], frame[1], frame[2], frame[3]))
        print("*******************************************************")
        print("Projectile Publish Stamp: {}".format(finish_stamp))
        print("Time SPENT in Perception: {}".format(finish_stamp - stamps[0]) )


if __name__ == "__main__":
    main()
