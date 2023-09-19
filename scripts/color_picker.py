#!/usr/bin/env python
import cv2
import numpy as np
# import open3d as o3d
import os
import pyzed.sl as sl
import rospy
from scipy.optimize import least_squares
from sensor_msgs.msg import PointCloud2, PointField
# from shield_planner_msgs.msg import Projectile
import std_msgs.msg
import sys
from visualization_msgs.msg import MarkerArray, Marker

import pdb

# Relative Imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"camera_calibration"))
from helpers import *
from constants import *

COUNT=0

def main():
    # Load image from file
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"camera_calibration/color_data/")
    og_path = img_path + "img_og_" + str(COUNT) + ".jpg"
    img = cv2.imread(og_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Code to pick color on-click on image
    def click_event(event, x, y,  flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print('Col: ', x, ' Row: ', y)
            print(hsv_img[y, x])
            
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',click_event)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    main()