#!/usr/bin/env python
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys, os
import pdb

from PIL import Image
from scipy.spatial.transform import Rotation as R

# Relative Imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"camera_calibration"))
from helpers import *
from constants import *

# Relevant Constants

# Checker board params
T_EEF_TO_CHECKERBOARD = np.eye(4)

# In this case, the ee pose is referring to transformation from base to tool0
# T_TOOL0_CONTROLLER_TO_TOOL0 = getTransformationMatrix([0, 0.0, -0.148], 
#                               R.from_euler("xyz", [0, 0, 0]).as_quat())
T_TOOL0_CONTROLLER_TO_TOOL0 = getTransformationMatrix([0, 0.0, 0], 
                              R.from_euler("xyz", [0, 0, 0]).as_quat())

# Num poses used for calibration
NUM_POSES = 100

class Calibration:
    def __init__(self, folder_name):
        
        self.ee_poses_ = []
        self.imgs_ = []

        # Camera params
        self.camera_intrinsic_matrix_ = CAMERA_INTRINSIC_MATRIX
        self.camera_distortion_matrix_ = CAMERA_DISTORTION_MATRIX
        
        self.T_eef_to_checkerboard = T_EEF_TO_CHECKERBOARD
        self.T_tool0_controller_to_tool_ = T_TOOL0_CONTROLLER_TO_TOOL0

        # Extrinsic matrix
        self.T_base_to_cam = []

        self.read_data(folder_name)

    def read_data(self, folder):
        '''
        Read data from folder, ee_pose will be given in txt file, where it is a Nx13 matrix.
        Each row is a row major order array of index + 3x1 translation vector + 3x3 rot mat flatten (row major) array.
        '''
        T_base_to_ee_all = np.loadtxt(os.path.join(folder, 'T_base_ee.txt'))
        T_ee_to_target = np.loadtxt(os.path.join(folder, 'T_ee_poi.txt'))
        
        # Setting end effector to checkerboard transformation
        ee_to_target_rot_mat = np.reshape(T_ee_to_target[3:], (3,3))
        ee_to_target_trans_vec = np.reshape(T_ee_to_target[0:3], (3,1))
        T_EEF_TO_CHECKERBOARD[0:3, 0:3] = ee_to_target_rot_mat
        T_EEF_TO_CHECKERBOARD[0, 3] = ee_to_target_trans_vec[0] * 1000
        T_EEF_TO_CHECKERBOARD[1, 3] = ee_to_target_trans_vec[1] * 1000
        T_EEF_TO_CHECKERBOARD[2, 3] = ee_to_target_trans_vec[2] * 1000


        for i in range(0, NUM_POSES):
            img_filename = os.path.join(folder, 'left/' +  str(i+1) + '.png')
            img = Image.open(img_filename)
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.imgs_.append(img)

            ee_pose_array = T_base_to_ee_all[i, :]
            ee_pose_rot_mat = np.reshape(ee_pose_array[4:], (3,3))
            ee_pose_trans_vec = np.reshape(ee_pose_array[1:4], (3,1))
            T_base_to_ee = np.eye(4)
            T_base_to_ee[0:3, 0:3] = ee_pose_rot_mat
            T_base_to_ee[0, 3] = ee_pose_trans_vec[0] * 1000 
            T_base_to_ee[1, 3] = ee_pose_trans_vec[1] * 1000
            T_base_to_ee[2, 3] = ee_pose_trans_vec[2] * 1000

            self.ee_poses_.append(fromTransformationMatrix(T_base_to_ee))
            
            

    def convert_to_tool_frame(self, ee_pose):
        T_base_to_tool_controller = getTransformationMatrix(ee_pose[0], ee_pose[1])
        T_base_to_tool = np.matmul(T_base_to_tool_controller, self.T_tool0_controller_to_tool_)

        # TF data is in metres
        T_base_to_tool[0, 3] = T_base_to_tool[0, 3] * 1000
        T_base_to_tool[1, 3] = T_base_to_tool[1, 3] * 1000
        T_base_to_tool[2, 3] = T_base_to_tool[2, 3] * 1000

        return fromTransformationMatrix(T_base_to_tool)


    def calibrate(self):
        obj_world_points = []
        obj_image_pts = []

        for (img, ee_pose) in zip(self.imgs_, self.ee_poses_):
            print(ee_pose)
            _ret = self.estimateCheckerboardPoseOnTarget(img, ee_pose)
            if _ret:
                obj_world_points.append(_ret[0])
                obj_image_pts.append(_ret[1])

        obj_world_points = np.vstack(obj_world_points)
        obj_image_pts = np.vstack(obj_image_pts)

        ret, rvec_world_cam, tvec_world_cam = \
            cv2.solvePnP(obj_world_points, obj_image_pts, \
                self.camera_intrinsic_matrix_, self.camera_distortion_matrix_)
        self.T_base_to_cam = \
            invPoseSE3(poseVecToTmat(
                        np.hstack([tvec_world_cam.reshape(-1), rvec_world_cam.reshape(-1)])))

        print(self.T_base_to_cam,'<== T_base_to_cam')


    def getBasetoCheckerboardPose(self, ee_pose):
        T_base_to_eef = getTransformationMatrix(ee_pose[0], ee_pose[1])
        T_base_to_checkerboard = np.matmul(T_base_to_eef, self.T_eef_to_checkerboard)

        # What is this rotation - this is to align checkerboard coordinate with image coordinate
        rotate_pi_by_2_Y_axis = getTransformationMatrix([0, 0.0, 0.0], 
                              R.from_euler("xyz", [0, -np.pi/2, 0]).as_quat()) 
        
        return  np.matmul(T_base_to_checkerboard, rotate_pi_by_2_Y_axis)


    def estimateCheckerboardPoseOnTarget(self, image, ee_pose, \
                                        checkerboard = (7,6), square_size = 14.):
        assert square_size == 14.
        assert checkerboard == (7,6)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((checkerboard[0]*checkerboard[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:checkerboard[0],0:checkerboard[1]].T.reshape(-1,2) * square_size
        axis = np.float32([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]).reshape(-1,3) * square_size

        ret_world_cam, corners_world_cam = cv2.findChessboardCorners(image, (checkerboard[0],checkerboard[1]),None)
        if ret_world_cam == True :
            corners2_world_cam = cv2.cornerSubPix(image,corners_world_cam,(11,11),(-1,-1),criteria)
        else:
            print('Checkerboard not found!')
            # plt.imshow(image)
            # plt.show()
            return None

        # Find the rotation and translation vectors.
        ret_cam_world, rvecs_cam_world, tvecs_cam_world = \
            cv2.solvePnP(objp, corners2_world_cam, \
                         self.camera_intrinsic_matrix_, \
                         self.camera_distortion_matrix_)

        # project 3D points to image plane
        imgpts_origin_world_cam, jac = \
            cv2.projectPoints(axis, rvecs_cam_world, \
                              tvecs_cam_world, \
                              self.camera_intrinsic_matrix_, \
                              self.camera_distortion_matrix_)

        # project the origin and the axes
        imgpts_all_world_cam, jac    = cv2.projectPoints(objp, rvecs_cam_world, \
                                        tvecs_cam_world, \
                                        self.camera_intrinsic_matrix_, \
                                        self.camera_distortion_matrix_)
        image_world_cam = drawAxes(image, imgpts_origin_world_cam)
        image_world_cam = cv2.drawChessboardCorners(image_world_cam, checkerboard, \
                                                    corners2_world_cam, ret_cam_world)
        for i in range(imgpts_all_world_cam.shape[0]):
            image_world_cam = cv2.circle(image_world_cam, \
                                        (int(imgpts_all_world_cam[i,0,0]), \
                                        int(imgpts_all_world_cam[i,0,1])), 15, (255,0,0))

        # plt.imshow(image_world_cam)
        # plt.show()

        T_base_to_checkerboard_on_target = self.getBasetoCheckerboardPose(ee_pose) # CHECKERBOARD ON TARGET
        T_world_camera_to_checkerboard_on_card = \
            poseVecToTmat(np.hstack([tvecs_cam_world.reshape(-1), \
                                     rvecs_cam_world.reshape(-1)]))
        # plotting utils
        T_checkerboard_on_target_to_base = T_base_to_checkerboard_on_target
        obj_pts_homog = np.ones((checkerboard[0]*checkerboard[1],4), dtype = np.float32)
        obj_pts_homog[:,0:3] = objp
        obj_pts_homog_transformed = np.matmul(T_checkerboard_on_target_to_base, obj_pts_homog.T)
        
        T_base_cam_world = np.matmul(T_checkerboard_on_target_to_base, \
                                    invPoseSE3(T_world_camera_to_checkerboard_on_card))

        rvec_base_to_world_cam, tvec_base_to_world_cam = \
            getRvecTvecFromSE3(invPoseSE3(T_base_cam_world))
        
        checkerboard_on_card_pts_img_seen_by_world_cam, jac = \
            cv2.projectPoints(obj_pts_homog_transformed[0:3, :], \
                              rvec_base_to_world_cam, tvec_base_to_world_cam, \
                              self.camera_intrinsic_matrix_, \
                              self.camera_distortion_matrix_)
        
        for i in range(checkerboard_on_card_pts_img_seen_by_world_cam.shape[0]):
            plt.plot(checkerboard_on_card_pts_img_seen_by_world_cam[i,0,0], \
                checkerboard_on_card_pts_img_seen_by_world_cam[i,0,1], marker = 'x', color = 'r', ms = 15)

        _ret = [obj_pts_homog_transformed[0:3, :].T , np.squeeze(corners2_world_cam)] 

        # plt.show()
        return _ret

    def test(self):
        SAMPLE_NUM = [1, 5, 12]

        for i in SAMPLE_NUM:
            img = self.imgs_[i]
            obj_world_points = self.test_obj_world_points(self.ee_poses_[i])
            img_pts = self.test_check_cam_pose(img, obj_world_points, plot=True)
            image_segmented = segmentBox(np.squeeze(img), np.squeeze(img_pts))
            plt.imshow(np.squeeze(image_segmented))
            plt.show()

    def test_obj_world_points(self, ee_pose):
        # broadcast robot info on demand 
        print('Points on target')
        target_points_target_coord = \
            np.array([[-14., -14.0, 0.,1], [98., -14.0, 0.,1], \
                [98.0, 84.0, 0.,1], [-14., 84.0, 0.,1]])
        
        T_base_to_checkerboard_on_target = self.getBasetoCheckerboardPose(ee_pose)
        points_world = np.matmul(T_base_to_checkerboard_on_target, target_points_target_coord.T)

        return points_world[0:3, :]


    def test_check_cam_pose(self, frame, world_points = None, plot = False):        
        rvec, tvec = getRvecTvecFromSE3(invPoseSE3(self.T_base_to_cam))
        points_cam = cv2.projectPoints(world_points, rvec, tvec, \
            self.camera_intrinsic_matrix_, self.camera_distortion_matrix_)[0]

        if plot:
            plt.clf()
            plt.imshow(frame)
            for i in range(points_cam.shape[0]):
                plt.plot(points_cam[i,0,0], points_cam[i,0,1],  marker = '.', color = 'r', ms = 1)
            plt.show()

        return points_cam

    def statistic(self):
        # Calculate the calibration error on every image
        for i in range(len(self.imgs_)):
            img = self.imgs_[i]
            obj_world_points = self.test_obj_world_points(self.ee_poses_[i])
            img_pts = self.test_check_cam_pose(img, obj_world_points)
            image_segmented = segmentBox(np.squeeze(img), np.squeeze(img_pts))
            plt.imshow(np.squeeze(image_segmented))
            plt.show()

    def base_to_cambase(self):
        # T_base_to_left_cam = CAMERA_EXTRINSIC_MATRIX_LEFT
        T_base_to_cambase = np.matmul(self.T_base_to_cam, T_LEFTOPT_TO_CAMBASE)
        R_base_to_cambase = R.from_matrix(T_base_to_cambase[0:3, 0:3])
        rot_base_to_cambase = R_base_to_cambase.as_euler('xyz')
        tran_base_to_cambase = T_base_to_cambase[0:3, 3]/1000
        print('Camera base pose in base frame (rpy, xyz): ', rot_base_to_cambase, tran_base_to_cambase)

def main():
    calibrator = Calibration('camera_calibration/calib_data/')
    calibrator.calibrate()
    # calibrator.statistic()
    calibrator.test()
    calibrator.base_to_cambase()


if __name__ == '__main__':
        main()