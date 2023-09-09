#!/usr/bin/env python
import cv2
import numpy as np
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
DIST_THRESHOLD=5
MIN_PIXEL=15
PUBLISH_PROJ=1

# min_radius = 1  # Minimum radius of the ball
# max_radius = 30  # Maximum radius of the ball

# Global Vars
Num_Frame = 6
measurements = []
stamps = []
finish_stamp = 0
pc_xyz_list = []
pc_rgb_list = []

# TODO: 
# - Fix point cloud 'rgb' field, currently it's bgr, not rgb

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

    # # Create ROS publisher for (x0, y0, z0, vx0, vy0, vz0)
    # rospy.init_node('zed_trajectory_publisher')
    # pub = rospy.Publisher('/zed/trajectory', Float64MultiArray, queue_size=10)

    # Capture images
    i = 0
    image= sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    confidence_map = sl.Mat()

    # mirror_ref = sl.Transform()
    # mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    # tr_np = mirror_ref.m

    count = 0
    t1 = None

    while i < 500:
        #A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            stamp_temp = rospy.Time.now().to_sec()
            i = i + 1
            fps = zed.get_current_fps()
            # parameters = zed.getruntime_parameters()
            # frame_rate = svo_parameters.get_svo_real_time_mode()
            print("Frame Rate: {} FPS".format(fps))
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            # zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)
            # continue

            # Convert ZED Mat objects to numpy arrays
            image_ocv = image.get_data()

            hsv_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2HSV)

            # Define the adjusted range for bright orange color in HSV
            lower_bound = np.array([8, 150, 150])
            upper_bound = np.array([20, 255, 255])

            # Create a binary mask for orange color in HSV
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            image_masked = cv2.bitwise_and(image_ocv, image_ocv, mask=mask)
            # cv2.namedWindow('image1')
            # cv2.imshow("image1", image_masked)
            # cv2.waitKey(0)
            where = np.where(mask == 255)
            if len(where[0]) > 0:
                if where[0].shape[0] < MIN_PIXEL:
                    continue
            # depth_np = depth.get_data()
            pc_xyz_np = point_cloud.get_data()[:,:,:3]
            pc_rgb_np = point_cloud.get_data()[:,:,3]
            # pc_rgb_np = cv2.cvtColor(pc_bgr_np, cv2.COLOR_BGR2RGB)
            # pdb.set_trace()
            # Transform xyz from camleft to base
            confidence_np = confidence_map.get_data()

            # depth_poi = depth_np[where[0], where[1]]
            confidence_poi_all = confidence_np[where[0], where[1]].reshape((-1,1))
            pc_xyz_poi_all = pc_xyz_np[where[0], where[1], :].reshape((-1,3))
            pc_rgb_poi_all = pc_rgb_np[where[0], where[1]].reshape((-1,1))
            valid_ind = np.logical_not(np.isnan(pc_xyz_poi_all[:,0]))

            confidence_poi = confidence_poi_all[valid_ind,:]
            pc_xyz_poi = pc_xyz_poi_all[valid_ind,:]
            pc_rgb_poi = pc_rgb_poi_all[valid_ind,:]
            pc_xyz_poi = np.dot(T_BASE_TO_LEFT, np.append(pc_xyz_poi, np.ones((pc_xyz_poi.shape[0],1)), axis=1).transpose())[0:3,:].transpose()

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

            # pdb.set_trace()
            # Apply Sobel filter for edge detection
            # sobel_image = cv2.Sobel(mask, cv2.CV_8U, 1, 0, ksize=3)
            # sobel_image = cv2.dilate(sobel_image, None, iterations=2)
            # sobel_image = cv2.erode(sobel_image, None, iterations=2)

            # cv2.namedWindow('image1')
            # cv2.imshow("image1", image_masked)
            # cv2.waitKey(0)

            # Find contours in the mask
            # contours, _ = cv2.findContours(sobel_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            # for contour in contours:

            # Fit a circle to the contour
            # (x, y), radius = cv2.minEnclosingCircle(contour)
            # center = (int(x), int(y))
            # radius = int(radius)

            # Apply radius threshold
            # if radius < min_radius or radius > max_radius:
            #     continue

            # Draw the bounding box
            # cv2.circle(image_ocv, center, radius, (0, 255, 0), 2)
            
            # Calculate centroid of the circle
            # centroid_x = int(x)
            # centroid_y = int(y)
            
            # err, point_cloud_value = point_cloud.get_value(centroid_x, centroid_y)

            # Calculate the mean of the depth
            mean_X = np.mean(pc_xyz_poi[:,0])
            print(mean_X)
            mean_Y = np.mean(pc_xyz_poi[:,1])
            mean_Z = np.mean(pc_xyz_poi[:,2])
            mean_Conf = np.mean(confidence_poi[:])
            # mean_Con = np.mean(confidence_poi)
            if mean_X != 0 and mean_X < DIST_THRESHOLD:
                # cv2.namedWindow('image1')
                # cv2.imshow("image1", image_masked)
                # cv2.waitKey(0)
                # pdb.set_trace()
                # exit()
                count = count + 1
                if count == 1:
                    t = 0.0
                    stamps.append(stamp_temp)
                else:
                    t = stamp_temp-stamps[0]
                    stamps.append(stamp_temp)
                # if t1 is None:
                #     t1 = rospy.Time.now().to_sec()
                #     t = 0.0  # t1 = 0 for the first set of measurements
                # else:
                #     # t = (rospy.Time.now() - t1).to_sec()  # Calculate t2 and t3 relative to t1
                #     t = (rospy.Time.now().to_sec() - t1)  # Calculate t2 and t3 relative to t1

                # print(f"t: {t}, X: {mean_X}, Y: {mean_Y}, Z: {mean_Z}, Confidence: {mean_Con}")
                print("t: {:.5f}, X: {:.5f}, Y: {:.5f}, Z: {:.5f}, Conf: {:.3f}".format(t, mean_X, mean_Y, mean_Z, mean_Conf))
                measurements.append((t, mean_X, mean_Y, mean_Z))

                # Store point cloud
                pc_xyz_list.append(pc_xyz_poi)
                pc_rgb_list.append(pc_rgb_poi)

            if count >= Num_Frame:
                # Perform trajectory estimation here using the measurements
                estimated_params = estimate_trajectory(measurements)
                print(f"Estimated Parameters (XYZ,VxVyVz): \n{estimated_params}")
                
                # # Publish (x0, y0, z0, vx0, vy0, vz0) as a ROS message
                projectile_msg = Projectile()
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'odom_combined'
                projectile_msg.header = header
                projectile_msg.object_id = 0

                # Setting the variable in the message to computed results
                projectile_msg.position.x = estimated_params[0]
                projectile_msg.position.y = estimated_params[1]
                projectile_msg.position.z = estimated_params[2]
                projectile_msg.velocity.x = estimated_params[3]
                projectile_msg.velocity.y = estimated_params[4]
                projectile_msg.velocity.z = estimated_params[5]

                if PUBLISH_PROJ:
                    projectile_msg_pub.publish(projectile_msg)
                    finish_stamp = rospy.Time.now().to_sec()
                    rospy.logwarn("Publishing projectile!")
                    print(projectile_msg)

                break

                # traj_msg = Float64MultiArray()
                # traj_msg.data = estimated_params
                # pub.publish(traj_msg)

                # measurements.clear()
                # t1 = None

            # if count == Num_Frame:
            #     break

            
            # def click_event(event, x, y,  flags, params):
            #     if event == cv2.EVENT_LBUTTONDBLCLK:
            #         print('Col: ', x, ' Row: ', y)
            #         print(hsv_image[y, x])
            
            # cv2.namedWindow('image')
            # cv2.setMouseCallback('image',click_event)
            # cv2.imshow("image", image_ocv)
            # cv2.waitKey(1)
            
            # point_cloud_np = point_cloud.get_data()
            # point_cloud_np.dot(tr_np)

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





if __name__ == "__main__":
    main()
