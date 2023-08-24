#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import sensor_msgs.point_cloud2 as pc2


min_radius = 1  # Minimum radius of the ball
max_radius = 100  # Maximum radius of the ball
Baseline = 0.12 # Unit is meter
focal_length = 500


def callback(rgb_msg, disparity_msg, pc_msg, confidence_msg):
    bridge = CvBridge()
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    disparity_image = bridge.imgmsg_to_cv2(disparity_msg, desired_encoding='passthrough')
    point_cloud = bridge.imgmsg_to_cv2(pc_msg, desired_encoding='passthrough')
    confidence_image = bridge.imgmsg_to_cv2(confidence_msg, desired_encoding='passthrough')

    # Convert ZED Mat objects to numpy arrays
    confidence_map_np = confidence_image
    disparity_map_np = disparity_image

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define the adjusted range for bright orange color in HSV
    lower_bound = np.array([8, 150, 150])
    upper_bound = np.array([20, 255, 255])

    # Create a binary mask for orange color in HSV
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_points = []

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        if radius < min_radius or radius > max_radius:
            continue
        
        # Draw the circle
        cv2.circle(rgb_image, center, radius, (0, 255, 0), 2)
        # x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate centroid of the circle
        centroid_x = int(x)
        centroid_y = int(y)

        # Get confidence and depth values
        confidence = confidence_map_np[centroid_x, centroid_y]
        
        disparity = disparity_map_np[centroid_x, centroid_y]

        depth = Baseline * focal_length / disparity

        if depth < 5 and confidence < 40:

            # Get the corresponding point cloud value at the centroid pixel
            point_cloud_value = point_cloud[centroid_x, centroid_y]

            # Append the filtered point with X, Y, Z coordinates
            filtered_points.append([point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]])

    # Publish the filtered point cloud
    if filtered_points:
        filtered_pc = pc2.create_cloud_xyz32(pc_msg.header, filtered_points)
        pub.publish(filtered_pc)

    cv2.imshow('RGB Image', rgb_image)
    cv2.waitKey(1)

def main():
    rospy.init_node('zed_subscriber', anonymous=True)
    pub = rospy.Publisher('/zed/filtered_point_cloud', PointCloud2, queue_size=10)

    # Subscribe to the RGB, Depth, Point Cloud, and Confidence topics using message_filters
    rgb_sub = message_filters.Subscriber('/zed/rgb/image_rect_color', Image)
    disparity_sub = message_filters.Subscriber('/zed/disparity/disparity_image', Image)
    point_cloud_sub = message_filters.Subscriber('/zed/point_cloud/cloud_registered', PointCloud2)
    confidence_sub = message_filters.Subscriber('/zed/confidence/confidence_map', Image)

    ts = message_filters.TimeSynchronizer([rgb_sub, disparity_sub, point_cloud_sub, confidence_sub], 10)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    main()
