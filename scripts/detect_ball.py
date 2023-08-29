#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import ros_numpy as rnp
import sensor_msgs.point_cloud2 as pc2

import pdb


min_radius = 3  # Minimum radius of the ball
max_radius = 100  # Maximum radius of the ball


def callback(rgb_msg, depth_msg, pc_msg, confidence_msg, pub):
    # print("In detect ball call back!!!")
    bridge = CvBridge()
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgra8')
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
    # point_cloud = bridge.imgmsg_to_cv2(pc_msg, desired_encoding='passthrough')
    point_cloud = rnp.numpify(pc_msg)
    confidence_image = bridge.imgmsg_to_cv2(confidence_msg, desired_encoding='32FC1')

    # Convert ZED Mat objects to numpy arrays
    depth_map_np = depth_image
    confidence_map_np = confidence_image

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define the adjusted range for bright orange color in HSV
    lower_bound = np.array([8, 150, 150])
    upper_bound = np.array([20, 255, 255])

    # Create a binary mask for orange color in HSV
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_points = []
    filtered_points_confident = []

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

        # Extract depth and confidence values
        # depth = depth_map_np[centroid_y,centroid_x]

        mask = np.zeros(rgb_image.shape, np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        where = np.where(mask == 255)
        depth_points = depth_map_np[where[0], where[1]]
        depth = np.mean(depth_points)

        # confidence = confidence_map_np[centroid_y,centroid_x]
        confidence = confidence_map_np[where[0],where[1]]
        confidence = np.mean(confidence, axis=0)

        if depth < 4 and confidence < 80:
            # Get the corresponding point cloud value at the centroid pixel
            # point_cloud_value = point_cloud[centroid_y,centroid_x]
            point_cloud_poi = point_cloud[where[0], where[1]] 
            poi = [[p[0], p[1], p[2]] for p in point_cloud_poi]
            poi = np.array(poi)
            # pdb.set_trace()

            point_cloud_value = np.mean(poi, axis=0)

            # Append the filtered point with X, Y, Z coordinates
            filtered_points.append([point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]])
            filtered_points_confident.append(confidence)

    # Publish the filtered point cloud
    if filtered_points:
        filtered_pc = pc2.create_cloud_xyz32(pc_msg.header, filtered_points)
        for ind, pc in enumerate(filtered_points):
            print("Publishing point_cloud_filtered {}: X:{:.2f}    Y:{:.2f}    Z:{:.2f}".format(ind, pc[0],pc[1],pc[2]))
            print("with confident:{:.2f}".format(filtered_points_confident[ind]))
        pub.publish(filtered_pc)

    # cv2.imshow('RGB Image', rgb_image)
    # cv2.waitKey(1)

def main():
    rospy.init_node('zed_subscriber', anonymous=True)
    pub = rospy.Publisher('/zed2i/filtered_point_cloud', PointCloud2, queue_size=10)

    # Subscribe to the RGB, Depth, Point Cloud, and Confidence topics using message_filters
    rgb_sub = message_filters.Subscriber('/zed2i/zed_nodelet/rgb/image_rect_color', Image)
    depth_sub = message_filters.Subscriber('/zed2i/zed_nodelet/depth/depth_registered', Image)
    point_cloud_sub = message_filters.Subscriber('/zed2i/zed_nodelet/point_cloud/cloud_registered', PointCloud2)
    confidence_sub = message_filters.Subscriber('/zed2i/zed_nodelet/confidence/confidence_map', Image)

    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, point_cloud_sub, confidence_sub], 10)
    ts.registerCallback(callback, pub)

    rospy.spin()

if __name__ == '__main__':
    main()
