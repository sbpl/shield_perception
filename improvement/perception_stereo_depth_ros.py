#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import ros_numpy as rnp
import sensor_msgs.point_cloud2 as pc2

# ZED Camera constants
# For HD720
Baseline = 0.12  # Use meter units
fx = 531.2548217773438
fy = 531.2548217773438
cx = 666.2540283203125
cy = 349.59100341796875

pc_msg = PointCloud2()  # Initialize the PointCloud2 message

def callback(rgb_left_msg, rgb_right_msg):
    global pc_msg  # Declare pub as a global variable

    bridge = CvBridge()
    rgb_left_image = bridge.imgmsg_to_cv2(rgb_left_msg, desired_encoding='bgra8')
    rgb_right_image = bridge.imgmsg_to_cv2(rgb_right_msg, desired_encoding='bgra8')
    
    hsv_left_image = cv2.cvtColor(rgb_left_image, cv2.COLOR_BGR2HSV)
    hsv_right_image = cv2.cvtColor(rgb_right_image, cv2.COLOR_BGR2HSV)

    # Create some variables to store centroid points
    left_centroid = None
    right_centroid = None

    filtered_points = []

    # Define the adjusted range for bright orange color in HSV
    lower_bound = np.array([8, 150, 150])
    upper_bound = np.array([20, 255, 255])

    # Create a binary mask for orange color in HSV
    mask_left = cv2.inRange(hsv_left_image, lower_bound, upper_bound)
    mask_right = cv2.inRange(hsv_right_image, lower_bound, upper_bound)

    # Find left contours in the mask
    contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_left:
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Apply radius threshold
        if 1 <= radius <= 30:
            left_centroid = center
            cv2.circle(rgb_left_image, center, radius, (0, 255, 0), 2)

    # Find right contours in the mask
    contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_right:
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Apply radius threshold
        if 1 <= radius <= 30:
            right_centroid = center
            cv2.circle(rgb_right_image, center, radius, (0, 255, 0), 2)

    # Check if centroids were detected in both views
    if left_centroid is not None and right_centroid is not None:
        left_x, left_y = left_centroid
        right_x, right_y = right_centroid

        # Calculate disparity for the detected points
        disparity = left_x - right_x

        # Calculate 3D global coordinates
        if disparity != 0:
            Z = Baseline * fx / disparity
            if 0.1 < Z < 5:
                X = Z * (left_x - cx) / fx 
                Y = Z * (left_y - cy) / fy 
                
                # Append the filtered point with X, Y, Z coordinates
                filtered_points.append([X, Y, Z])

    # Publish the filtered point cloud
    if filtered_points:
        pc_data = np.array(filtered_points, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
        pc_msg = pc2.create_cloud_xyz32(pc_msg.header, pc_data)

def main():
    global pc_msg  # Declare pc_msg as a global variable
    rospy.init_node('zed_subscriber', anonymous=True)
    pub = rospy.Publisher('/zed/filtered_point_cloud', PointCloud2, queue_size=10)

    # Subscribe to the RGB topics using message_filters
    rgb_left_sub = message_filters.Subscriber('/zed2i/zed_nodelet/rgb/image_rect_color', Image)
    rgb_right_sub = message_filters.Subscriber('/zed2i/zed_nodelet/right/image_rect_color', Image)

    ts = message_filters.TimeSynchronizer([rgb_left_sub, rgb_right_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    main()