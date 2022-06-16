#!/usr/bin/env python

import numpy as np 
# import rospkg
from os import listdir
from os.path import isfile, join
from scipy.spatial.transform import Rotation as R

# rospack = rospkg.RosPack()

def get_skm(v):
       return np.array([[0,-v[2],v[1]],
                        [v[2],0,-v[0]],
                        [-v[1],v[0],0]])

def ComputeRotationMSO(a, b):
       v = np.cross(a,b)
       s = np.linalg.norm(v)
       c = np.dot(a,b)
       Rot = np.eye(3) + get_skm(v) + np.dot(get_skm(v), get_skm(v)) * (1.0/(1.0+c))
       return Rot

if __name__ == '__main__':
    
    # log_path = rospack.get_path('shield_perception')+"/logs/"
    log_path = "/home/roman/catkin_ws/src/shield_planner/shield_perception/logs/"

    log_files = [f for f in listdir(log_path) if isfile(join(log_path, f))]
    # log_files = ['/home/roman/catkin_ws/src/shield_planner/shield_perception/logs/projectile_throw_bag_7_p5.npy']

    rotated_vectors = np.zeros((len(log_files), 3))
    z_up = np.array([[0],[0],[1]])

    for i in range(len(log_files)):
        file_name = log_files[i]
        a = np.load(join(log_path, file_name))
        SC_vel = np.array(a[1,:])
        KI_vel = np.array(a[3,:])
        print("SC_vel.shape ", SC_vel.shape)
        print("KI_vel.shape ", KI_vel.shape)
        # SC_vel[2] = 0.0
        # KI_vel[2] = 0.0

        KI_vel_dir = KI_vel/np.linalg.norm(KI_vel)
        SC_vel_dir = SC_vel/np.linalg.norm(SC_vel)
        print("velocities", np.linalg.norm(KI_vel), np.linalg.norm(SC_vel))
        # print("Yaw", np.arccos(np.dot(KI_vel_dir,SC_vel_dir))*180/np.pi)

        R_SC_KI = ComputeRotationMSO(SC_vel_dir, KI_vel_dir)
        rotated_vectors[i,:] = np.dot(R_SC_KI, z_up)[:,0]
        print("For file:", i, file_name)
        print("SC_VEL", "->", SC_vel_dir)
        print("KI_vel", "->", KI_vel_dir)
        print("Rotation vector", "->", rotated_vectors[i,:])
        r = R.from_matrix(R_SC_KI).as_euler('zxy', degrees=True)
        print("Euler Angles", "->", r)
        print('\n\n\n')
    
    combined_rotation_vector = np.mean(rotated_vectors,axis=0)
    combined_rotation_vector = combined_rotation_vector / np.linalg.norm(combined_rotation_vector)
    print("Combined vector", i, "->", combined_rotation_vector)
    R_SC_KI = ComputeRotationMSO(z_up[:,0], combined_rotation_vector)
    print("Estimated Rotation:", R_SC_KI)

    translation_estimates = np.zeros((len(log_files), 3))
    for i in range(len(log_files)):
        file_name = log_files[i]
        a = np.load(join(log_path, file_name))
        SC_pos = np.array(a[0,:])
        SC_vel = np.array(a[1,:])
        KI_pos = np.array(a[2,:])
        KI_vel = np.array(a[3,:])

        T_SC_KI = KI_pos.reshape((-1,1)) - np.dot(R_SC_KI, SC_pos.reshape((-1,1)))
        translation_estimates[i,:] = T_SC_KI[:,0]
        print("Translation vector", i, "->", translation_estimates[i,:])
        KI2_vel = np.dot(R_SC_KI, KI_vel.reshape((-1,1)))
        KI2_pos = np.dot(R_SC_KI, SC_pos.reshape((-1,1))) + T_SC_KI.reshape((-1,1))
        print("Verifications", KI_pos, KI2_pos, KI_vel, KI2_vel)
    
    T_SC_KI = np.mean(translation_estimates,axis=0).reshape((-1,1))
    print("Estimated translation", T_SC_KI)

    save_dir = "/home/roman/catkin_ws/src/shield_planner/shield_perception/"
    np.save(save_dir+"/R_SC_KI.npy", R_SC_KI)
    np.save(save_dir+"/T_SC_KI.npy", T_SC_KI)
