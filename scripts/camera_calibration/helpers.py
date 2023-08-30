#!/usr/bin/env python

import cv2
import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R
import numpy.matlib as npm

def drawAxes(img, imgpts):
    if len(img.shape) == 2:
        img = np.dstack([img]*3)
    imgpts = np.array(imgpts).astype('int')
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,255,0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
    return img

def poseVecToTmat(posevec):
    # make T mat from posevec
    r_rotvec = R.from_rotvec([posevec[3], posevec[4], posevec[5]])
    r_rotmat = r_rotvec.as_matrix()
    T = np.eye(4)
    T[0:3,0:3] = r_rotmat
    T[0, 3] = posevec[0]
    T[1, 3] = posevec[1]
    T[2, 3] = posevec[2]
    return T

def getTransformationMatrix(position, quat):
    r = R.from_quat(quat)
    T = np.eye(4)
    T[0:3, 0:3] = r.as_matrix()
    T[0, 3] = position[0]
    T[1, 3] = position[1]
    T[2, 3] = position[2]

    return T

def fromTransformationMatrix(T):
    q = R.from_matrix(T[:3, :3]).as_quat()
    position = [T[0, 3], T[1, 3], T[2, 3]]
    return position, q

def invertTransformationMatrix(T):
    position = T[:3, 3]
    rot = T[:3, :3]
    inverted_rot = R.from_matrix(rot).inv().as_matrix()
    inverted_position = -np.matmul(inverted_rot, position)
    
    inverted_trans = np.eye(4)
    inverted_trans[:3, :3] = inverted_rot
    inverted_trans[:3, 3] = inverted_position

    return inverted_trans

def convertToMM(T):
    # Converts from Metres to MM
    T[0, 3] = T[0, 3] * 1000
    T[1, 3] = T[1, 3] * 1000
    T[2, 3] = T[2, 3] * 1000

    return T

def invPoseSE3(T):
    assert T.shape == (4,4)
    T_inv = np.eye(4)
    R = T[0:3, 0:3]
    R_inv = R.T
    t = T[0:3, -1].reshape(3,1)
    t_inv = np.matmul(-R_inv, t)
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, -1] = t_inv.reshape(3,)
    return T_inv    

def getRvecTvecFromSE3(T):
    ''' returns rvec, tvec from T'''
    R = T[0:3, 0:3]
    rvec = cv2.Rodrigues(R)[0]
    tvec = T[0:3, -1]
    return rvec.reshape(3,), tvec

def testPoints(ax, ay, bx, by, x1, y1, x2,y2):
    return ((y1 - y2)*(ax - x1) + (x2 - x1)*(ay - y1))*((y1 - y2)*(bx - x1) + (x2 - x1)*(by - y1)) < 0.

def segmentBox(image, corner_point):

    X, Y = np.meshgrid(np.arange(0, image.shape[1], 1, dtype = float ), np.arange(0, image.shape[0],1, dtype= float)) 
    X_ = X.flatten()
    Y_ = Y.flatten()
    _ones = np.ones_like(X_)
    centroid_x = (corner_point[:,0].mean())* _ones
    centroid_y = (corner_point[:,1].mean())* _ones

    point_1_x = corner_point[0,0] * _ones
    point_2_x = corner_point[1,0] * _ones
    point_3_x = corner_point[2,0] * _ones
    point_4_x = corner_point[3,0] * _ones

    point_1_y = corner_point[0,1] * _ones
    point_2_y = corner_point[1,1] * _ones
    point_3_y = corner_point[2,1] * _ones
    point_4_y = corner_point[3,1] * _ones

    test_point_1_to_2 = testPoints(X_, Y_, centroid_x, centroid_y, point_1_x, point_1_y, point_2_x, point_2_y)
    test_point_2_to_3 = testPoints(X_, Y_, centroid_x, centroid_y, point_2_x, point_2_y, point_3_x, point_3_y)
    test_point_3_to_4 = testPoints(X_, Y_, centroid_x, centroid_y, point_3_x, point_3_y, point_4_x, point_4_y)
    test_point_4_to_1 = testPoints(X_, Y_, centroid_x, centroid_y, point_4_x, point_4_y, point_1_x, point_1_y)


    mask = ~ test_point_1_to_2 & ~ test_point_2_to_3 & ~ test_point_3_to_4 & ~ test_point_4_to_1
    image_masked = image.flatten() * mask.astype(float)

    return image_masked.reshape(image.shape)

def averageQuaternions(Q):
    # Takes in quaternions in scalar last format and returns a quaternion
    # in the same format

    # Converting quaternions such that scalar is in front
    Q[: , [2, 3]] = Q[: , [3, 2]]
    Q[: , [1, 2]] = Q[: , [2, 1]]
    Q[: , [0, 1]] = Q[: , [1, 0]]

    # Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
    # The quaternions are arranged as (w,x,y,z), with w being the scalar
    # The result will be the average quaternion of the input. Note that the signs
    # of the output quaternion can be reversed, since q and -q describe the same orientation


    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    quat = np.real(eigenVectors[:,0].A1)

    # Converting quaternions such that scalar is in the last
    quat[[0, 1]] = quat[[1, 0]]
    quat[[1, 2]] = quat[[2, 1]]
    quat[[2, 3]] = quat[[3, 2]]

    return quat
