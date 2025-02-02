import cv2
import numpy as np
import math
print(cv2.__version__)

CONFIDENCE_VAL = 0.75

def color_filter(image_ocv, VISUAL):
    """
    COLOR FILTER implementation.
    """
    hsv_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2HSV)
    # Define the adjusted range for bright orange color in HSV
    lower_bound_orange = np.array([5, 75, 100])
    upper_bound_orange = np.array([30, 255, 255])
    # Define the adjusted range for green color in HSV
    lower_bound_green = np.array([50, 75, 100])
    upper_bound_green = np.array([85, 255, 255])

    # Create a binary mask for orange color in HSV
    mask_orange = cv2.inRange(hsv_image, lower_bound_orange, upper_bound_orange)
    mask_green = cv2.inRange(hsv_image, lower_bound_green, upper_bound_green)
    mask = mask_orange + mask_green
    # mask = mask_orange
    where = np.where(mask == 255)
    # DEBUG: Visualize masked image
    if VISUAL:
        image_masked = cv2.bitwise_and(image_ocv, image_ocv, mask=mask)
        cv2.namedWindow('image1')
        cv2.imshow("image1", image_masked)
        cv2.waitKey(1)
    return where

def conversion_bbox_mask(x1,x2,y1,y2):
    """
    Convert bbox corners to the same format as color filter 'where' variable.
    """
    # y is row, x is column
    y_array = np.array([i for i in range(y1, y2+1)])
    x_array = np.array([i for i in range(x1, x2+1)])
    return (y_array, x_array)

def reverse_map_sigmoid(x, input_min=0.000395, input_max=0.01, output_min=0.8, output_max=0.2):
    # Normalize input
    normalized_x = (x - input_min) / (input_max - input_min)
    # Sigmoid function with steepness k=10 and midpoint a=0.3 (normalized space)
    sigmoid = output_min + (output_max - output_min) / (1 + np.exp(-100 * (normalized_x - 0.3)))
    return sigmoid

def shrink_bbox_area(frame_size, x1,x2,y1,y2, vary=True, r=0.2):
    '''
    shrink x and y axis with ratio r.
    '''
    frame_area = frame_size[0]*frame_size[1]
    width = x2-x1
    height = y2-y1
    area_ratio = width*height/frame_area
    if vary:
        # Change ratio value to modify bbox size
        r = reverse_map_sigmoid(area_ratio)
    #print("r", r)
    subtraction_width = math.ceil((1-math.sqrt(r))/2*width)
    subtraction_height = math.ceil((1-math.sqrt(r))/2*height)

    x1_new = subtraction_width + x1
    x2_new = x2 - subtraction_width
    y1_new = subtraction_height + y1
    y2_new = y2 - subtraction_height

    return x1_new, x2_new, y1_new, y2_new

def retrained_YOLOv8(image, model, VISUAL):
    """
    RETRAINED YOLOv8 IMPLEMENTATION.

    Run retrained YOLOv8 detection. 
    The model only detect person and ball.
    Right now the function follows the color filter result 
    to generate output which contains x and y in each column respectively.
    This is for testing easily at the early stage.
    Later it will be changed to its own method.
    """
    map = {0:'person', 1: 'ball'}
    results = model(image)
    df = results[0].boxes
    bboxs = df.xyxy.cpu().numpy()
    labels = df.cls.cpu().numpy()  
    scores = df.conf.cpu().numpy()
    x1, x2, y1, y2 = None, None, None, None
    for p in range(len(bboxs)):
        if scores[p] > CONFIDENCE_VAL and labels[p] ==1:
            x1, y1, x2, y2 = int(bboxs[p][0]), int(bboxs[p][1]), int(bboxs[p][2]), int(bboxs[p][3])
            #center_x = (x1 + x2) // 2 # in pixel unit
            #center_y = (y1 + y2) // 2
            
            if VISUAL:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label_name = map[labels[p]]
                cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                image = cv2.resize(image, (960, 540))  
                cv2.imshow("Retrained YOLOv8 ", image)
    return x1,x2,y1,y2

def center_xyz(list_,r = 0.06):
    """
    Calculate the center of the projectile.
    Parameters:
        r: The radius of the projectile is 0.06m.
        list_: list of points in xyz.
    """
    yaw = math.atan2(list_[1],list_[0])
    pitch = math.atan2(list_[2], math.sqrt(list_[0]**2 + list_[1]**2))
    updated_x = list_[0] + r * math.cos(pitch) * math.cos(yaw)  # Shift along X-axis
    updated_y = list_[1] + r * math.cos(pitch) * math.sin(yaw)  # Shift along Y-axis
    updated_z = list_[2] + r * math.sin(pitch) 
    return [updated_x, updated_y, updated_z]
