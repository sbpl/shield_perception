import cv2
import numpy as np
import sys
import pyzed.sl as sl
import math
from numpy.fft import fft2, ifft2

min_radius = 1  # Minimum radius of the ball
max_radius = 20  # Maximum radius of the ball


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

    # Capture images
    i = 0
    image= sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    confidence_map = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m


    count=0

    while i < 3000:
        #A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

            # Convert ZED Mat objects to numpy arrays
            image_ocv = image.get_data()

            hsv_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2HSV)

            # Define the adjusted range for bright orange color in HSV
            lower_bound = np.array([8, 150, 150])
            upper_bound = np.array([15, 255, 255])

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
                # x, y, w, h = cv2.boundingRect(contour)
                # cv2.rectangle(image_ocv, (x, y), (x + w, y + h), (0, 0, 255), 2)
  
                # Calculate centroid of the circle
                centroid_x = int(x)
                centroid_y = int(y)
                
                err, point_cloud_value = point_cloud.get_value(centroid_x, centroid_y)
                
                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                # distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                #                     point_cloud_value[1] * point_cloud_value[1] +
                #                     point_cloud_value[2] * point_cloud_value[2])


                if  point_cloud_value[2] < 5:
                    count = count + 1
                    print (point_cloud_value[0],point_cloud_value[1],point_cloud_value[2],confidence_map.get_value(centroid_x, centroid_y))
            
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
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
