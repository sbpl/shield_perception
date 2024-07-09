#!/usr/bin/env python
import cv2
import numpy as np
import os
import pickle
import pyzed.sl as sl

def main():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps=60
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    # init.depth_stabilization=50
    init.coordinate_units = sl.UNIT.METER # Use meter units (for depth measurements)
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    # Img
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.texture_confidence_threshold = 100
    if cam.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        image= sl.Mat()
        cam.retrieve_image(image, sl.VIEW.LEFT)
        image_ocv = image.get_data()
        cv2.namedWindow('image1')
        cv2.imshow("image1", image_ocv)
        cv2.waitKey(0)


    cam_sets = {}
    for s in sl.VIDEO_SETTINGS:
        # print(s, s.value)
        # print(s, cam.get_camera_settings(s)[1])
        cam_sets[s] = cam.get_camera_settings(s)[1]
        # if s == sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO:
        #     cam_sets[s] = 0

    print(cam_sets)
    with open('./cam_settings_data/lights_ceil.pickle', "wb") as cam_set_file:
        pickle.dump(cam_sets, cam_set_file)


if __name__ == '__main__':
    main()
