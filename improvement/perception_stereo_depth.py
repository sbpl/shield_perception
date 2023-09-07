import rospy
import cv2
import numpy as np
import sys
import pyzed.sl as sl
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


# ZED Camera constants
# For HD720
Baseline = 0.12  # Use meter units
fx = 531.2548217773438
fy = 531.2548217773438
cx = 666.2540283203125
cy = 349.59100341796875

# Simulated measurement data (time, x, y, z)
measurements = []

def main():
    # Init a rospy node
    rospy.init_node('zed_perception')

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2:
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER  # Use meter units

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 50
    runtime_parameters.texture_confidence_threshold = 100

    # Create some variables to store centroid and depth information
    left_centroid = None
    right_centroid = None

    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left and right images
            image_sl_left = sl.Mat()
            zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
            image_cv_left = image_sl_left.get_data()

            image_sl_right = sl.Mat()
            zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
            image_cv_right = image_sl_right.get_data()

            hsv_left_image = cv2.cvtColor(image_cv_left, cv2.COLOR_BGR2HSV)
            hsv_right_image = cv2.cvtColor(image_cv_right, cv2.COLOR_BGR2HSV)

            # Define the range for orange color in HSV
            lower_bound = np.array([10, 150, 150])
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
                    cv2.circle(image_cv_left, center, radius, (0, 255, 0), 2)
                    t = rospy.Time.now()
                    measurements.append((t, x, y, 0.0))

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
                    cv2.circle(image_cv_right, center, radius, (0, 255, 0), 2)

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
                        measurements.append((X, Y, Z))

                    # Perform trajectory estimation here using the measurements
                    estimated_params = estimate_trajectory(measurements)


            # Display the images
            cv2.imshow("Left Image", image_cv_left)
            cv2.imshow("Right Image", image_cv_right)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
