#!/usr/bin/env python

import cv2
import numpy as np
import sys
import pyzed.sl as sl
from numpy.fft import fft2, ifft2
import rospy
from scipy.optimize import least_squares
from std_msgs.msg import Float64MultiArray

min_radius = 1  # Minimum radius of the ball
max_radius = 30  # Maximum radius of the ball

Num_Frame = 3
measurements = []

def main() :
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER # Use meter units (for depth measurements)

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

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    count = 0
    t1 = None

    while i < 100:
        #A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            i = i + 1
            # SVO file frame rate
            fps = zed.get_current_fps()
            print("Frame Rate: {} FPS".format(fps))
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            # zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            # zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)
            # continue
            # Convert ZED Mat objects to numpy arrays
            image_ocv = image.get_data()

            hsv_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2HSV)

            # Define the adjusted range for bright orange color in HSV
            lower_bound = np.array([8, 150, 150])
            upper_bound = np.array([20, 255, 255])

            # Create a binary mask for orange color in HSV
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            

            # Apply Sobel filter for edge detection
            sobel_image = cv2.Sobel(mask, cv2.CV_8U, 1, 0, ksize=3)
            sobel_image = cv2.dilate(sobel_image, None, iterations=2)
            sobel_image = cv2.erode(sobel_image, None, iterations=2)


            # Find contours in the mask
            contours, _ = cv2.findContours(sobel_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            

            for contour in contours:

                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Apply radius threshold
                if radius < min_radius or radius > max_radius:
                    continue

                # Draw the bounding box
                cv2.circle(image_ocv, center, radius, (0, 255, 0), 2)
                
                # Calculate centroid of the circle
                centroid_x = int(x)
                centroid_y = int(y)
                
                err, point_cloud_value = point_cloud.get_value(centroid_x, centroid_y)


                if point_cloud_value[2] < 5:
                    count = count + 1
                    if t1 is None:
                        t1 = rospy.Time.now()
                        t = 0.0  # t1 = 0 for the first set of measurements
                    else:
                        t = (rospy.Time.now() - t1).to_sec()  # Calculate t2 and t3 relative to t1

                    print(f"t: {t}, X: {point_cloud_value[0]}, Y: {point_cloud_value[1]}, Z: {point_cloud_value[2]}, Confidence: {confidence_map.get_value(centroid_x, centroid_y)}")
                    measurements.append((t, point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]))

                if count >= Num_Frame:
                    # Perform trajectory estimation here using the measurements
                    estimated_params = estimate_trajectory(measurements)
                    print(f"Estimated Parameters: {estimated_params}")
                    
                    # # Publish (x0, y0, z0, vx0, vy0, vz0) as a ROS message
                    # traj_msg = Float64MultiArray()
                    # traj_msg.data = estimated_params
                    # pub.publish(traj_msg)

                    measurements.clear()
                    t1 = None

            
            def click_event(event, x, y,  flags, params):
                if event == cv2.EVENT_LBUTTONDBLCLK:
                    print('Col: ', x, ' Row: ', y)
                    print(hsv_image[y, x])
            
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',click_event)
            cv2.imshow("image", image_ocv)
            cv2.waitKey(0)
            
            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)

            i = i + 1

    cv2.destroyAllWindows()
    zed.close()

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


if __name__ == "__main__":
    main()
