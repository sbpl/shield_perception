#! /usr/bin/env python

import os
import mujoco as mp
from mujoco import MjData, MjModel
import mujoco_viewer
from time import sleep
import numpy as np
import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
import pickle
import math
# zed
import cv2
import pyzed.sl as sl

# ROS
import rospy
import actionlib
from pydrake.all import BsplineTrajectory, KinematicTrajectoryOptimization, Solve
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState

sim = True

# model
model_dir = '/home/shield/code/parallel_search/third_party/mujoco-2.3.2/model/abb/irb_1600/'
mjcf = 'irb1600_6_12_camcalib.xml'

# viewer params
viz_dt = 1

# dm_control
manip_joints = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
# calib_target_ori_range = np.array([[np.pi, 0, -np.pi/4], [np.pi/6, np.pi/6, np.pi/6]])
calib_target_ori_range = np.array([[3.098, 0.692, -0.850], [np.pi/6, np.pi/6, np.pi/6]])
model = MjModel.from_xml_path(os.path.join(model_dir, mjcf))
data = MjData(model)
phy = mujoco.Physics.from_xml_path(os.path.join(model_dir, mjcf))
if sim:
    viewer = mujoco_viewer.MujocoViewer(model, data)

# camera settings
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD2K
init_params.camera_fps = 30
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(-1)

# output save dir
num_samples = 100
save_dir = '/home/shield/code/shield_ws/src/shield_perception/scripts/camera_calibration/calib_data'

# planner params
nq = 6
qmin = np.array([-3.14159, -1.0995, -4.1015, -3.4906, -2.0071, -6.9813])
qmax  = np.array([3.14159, 1.9198, 0.9599, 3.4906, 2.0071, 6.9813])
dqmin = np.array([-2.618, -2.7925, -2.967, -5.585, -6.9813, -7.854])
dqmax = np.array([2.618, 2.7925, 2.967, 5.585, 6.9813, 7.854])
ddqmin = -10*np.ones(nq)
ddqmax = 10*np.ones(nq)
dddqmin = -100*np.ones(nq)
dddqmax = 100*np.ones(nq)
dt = 4e-3


# ROS subscriber
curr_state = JointTrajectoryControllerState()
# curr_state.actual.positions = np.zeros(nq) # initialize with zeros
# def robotStateCallback(data):
#     curr_state = data

# execution
with open('/home/shield/code/shield_ws/src/abb_robot_driver/abb_robot_bringup_examples/scripts/calib_start.pkl', 'rb') as file:
    start_tpva = pickle.load(file)

with open('/home/shield/code/shield_ws/src/abb_robot_driver/abb_robot_bringup_examples/scripts/calib_end.pkl', 'rb') as file:
    end_tpva = pickle.load(file)
calib_end_start = np.array([0, 1.57, -1.57, 0, 0, 0])

def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]

def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def getCalibTarget(m, d):
    id = mp.mj_name2id(m, mp.mjtObj.mjOBJ_SITE, 'calib_target')
    calib_target = [d.site_xpos[id], m.site_size[id]]
    return calib_target

def sampleFromCalibTarget(calib_target):
    seed = np.random.random_sample((3,))
    offset = 2*calib_target[1]*seed - calib_target[1]
    sample_xpos = calib_target[0] + offset

    ori_seed = np.random.random_sample((3,))
    ori_offset = 2*calib_target_ori_range[1]*ori_seed - calib_target_ori_range[1]
    sample_rpy = calib_target_ori_range[0] + ori_offset
    sample_xquat = get_quaternion_from_euler(sample_rpy[0], sample_rpy[1], sample_rpy[2])
    return [sample_xpos, sample_xquat]

def sampleCalibrationPose(calib_target):
    s = sampleFromCalibTarget(calib_target)
    ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
    for i in range(100):
        if ik_result.success:
            break
        s = sampleFromCalibTarget(calib_target)
        ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
    return ik_result.qpos

def findLocalTransform(T_W_A_pos, T_W_A_quat, T_W_B_pos, T_W_B_quat):
    T_W_A_pos_inv = np.zeros(3)
    T_W_A_quat_inv = np.zeros(4)
    mp.mju_negPose(T_W_A_pos_inv, T_W_A_quat_inv, T_W_A_pos, T_W_A_quat)

    T_A_B_pos = np.zeros(3)
    T_A_B_quat = np.zeros(4)
    mp.mju_mulPose(T_A_B_pos, T_A_B_quat, T_W_A_pos_inv, T_W_A_quat_inv, T_W_B_pos, T_W_B_quat)
    return T_A_B_pos, T_A_B_quat

def getSiteTransformWRTWorld(site_name):
    id = mp.mj_name2id(model, mp.mjtObj.mjOBJ_SITE, site_name)
    T_W_site_pos = data.site_xpos[id]
    T_W_site_quat = np.zeros(4)
    mp.mju_mat2Quat(T_W_site_quat, data.site_xmat[id])
    return T_W_site_pos, T_W_site_quat


def generateTrajectory(q0, qF):
    wp = np.array([])
    trajopt = KinematicTrajectoryOptimization(nq, 10, 4)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(10.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(qmin, qmax)
    trajopt.AddVelocityBounds(dqmin, dqmax)
    trajopt.AddAccelerationBounds(ddqmin, ddqmax)
    trajopt.AddJerkBounds(ddqmin, ddqmax)
    trajopt.AddDurationConstraint(0.5, 25)
    trajopt.AddPathPositionConstraint(q0, q0, 0)
    trajopt.AddPathPositionConstraint(qF, qF, 1)
    prog.AddQuadraticErrorCost(np.eye(nq), q0, trajopt.control_points()[:, -1])
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 0) # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((nq, 1)), np.zeros((nq, 1)), 1) # start and end with zero velocity

    ds = 1.0/(np.shape(wp)[0]+1)
    for i, r in zip(range(np.shape(wp)[0]), wp):
        trajopt.AddPathPositionConstraint(r, r, (i+1)*ds)

    # Solve once without the collisions and set that as the initial guess for
    # the version with collisions.
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed, even without collisions!")
    # print("trajopt succeeded!")
    op_traj = trajopt.ReconstructTrajectory(result)
    # print('traj duration: ', op_traj.end_time())

    tpva = np.empty((int(np.ceil(op_traj.end_time()/dt)), 1+3*nq))
    test_goal = FollowJointTrajectoryGoal()
    goal_traj = JointTrajectory()
    goal_traj.header.frame_id = 'odom_combined'
    goal_traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    j = 0
    for t in np.arange(op_traj.start_time(), op_traj.end_time(), dt):
        p = JointTrajectoryPoint()
        p.time_from_start.secs = int(t)
        p.time_from_start.nsecs = int(rospy.Time.from_sec(t).to_nsec() % 1e9)
        p.positions = [0,0,0,0,0,0]
        p.velocities = [0,0,0,0,0,0]
        p.accelerations = [0,0,0,0,0,0]

        tpva[j, 0] = t

        for i in range(nq):
            p.positions[i] = op_traj.value(t)[i][0]
            # p.velocities[i] = op_traj.EvalDerivative(t, 1)[i][0]
            # p.accelerations[i] = op_traj.EvalDerivative(t, 2)[i][0]

            tpva[j, 1+i] = op_traj.value(t)[i][0]
            tpva[j, 1+nq+i] = op_traj.EvalDerivative(t, 1)[i][0]
            tpva[j, 1+2*nq+i] = op_traj.EvalDerivative(t, 2)[i][0]

        j += 1

    return tpva


def isValidTrajectory(traj):
    for wp in traj:
        for i in range(model.nq):
            data.qpos[i] = wp[1+i]

    mp.mj_forward(model, data)

    if data.ncon > 0:
        return True
    else:
        return True

def updateCurrentState():
    global curr_state
    curr_state = rospy.wait_for_message("/egm/joint_velocity_trajectory_controller/state", JointTrajectoryControllerState)
    data.qpos[:] = curr_state.actual.positions[:]
    mp.mj_kinematics(model, data)

def executeTraj(tpva):
    duration = tpva[-1][0]

    test_goal = FollowJointTrajectoryGoal()
    goal_traj = JointTrajectory()
    goal_traj.header.frame_id = 'odom_combined'
    goal_traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    for r in tpva:
        p = JointTrajectoryPoint()
        p.time_from_start.secs = int(r[0])
        p.time_from_start.nsecs = int(rospy.Time.from_sec(r[0]).to_nsec() % 1e9)
        p.positions = [0,0,0,0,0,0]
        p.velocities = [0,0,0,0,0,0]
        p.accelerations = [0,0,0,0,0,0]

        for i in range(nq):
            p.positions[i] = r[1+i]
            # p.velocities[i] = r[1+nq+i]
            # p.accelerations[i] = r[1+2*nq+i]
        goal_traj.points.append(p)

    test_goal.trajectory = goal_traj
    client.send_goal_and_wait(test_goal)
    client.wait_for_result(rospy.Duration.from_sec(duration+2))
    sleep(duration+1)

    # curr_state.actual.positions = tpva[-1][1:model.nq+1]
    updateCurrentState()

def visualize(tpva):
    viz_dt = dt
    if viewer.is_alive:
        for wp in tpva:
            data.qpos[:] = wp[1:model.nq+1]
            mp.mj_kinematics(model, data)
            viewer.render()
            sleep(viz_dt)
    curr_state.actual.positions = tpva[-1][1:model.nq+1]
    sleep(0.5)

def clickAndSavePic(id):
    limg = sl.Mat()
    rimg = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        lret = zed.retrieve_image(limg, sl.VIEW.LEFT) # Retrieve the left image
        rret = zed.retrieve_image(rimg, sl.VIEW.RIGHT) # Retrieve the left image
        if lret == sl.ERROR_CODE.SUCCESS and rret == sl.ERROR_CODE.SUCCESS:
          f = os.path.join(save_dir, 'left', str(id)+'.png')
          limg.write(f)
          f = os.path.join(save_dir, 'right', str(id)+'.png')
          rimg.write(f)

if __name__ == "__main__":

    rospy.init_node('camcalib_node')
    # rospy.Subscriber("/egm/joint_velocity_trajectory_controller/state", JointTrajectoryControllerState, robotStateCallback)
    client = actionlib.SimpleActionClient('egm/joint_velocity_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    # client = actionlib.SimpleActionClient('egm/joint_position_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration.from_sec(1.0)):
        print("joint_trajectory_action server not available")
    print("Connected to follow_joint_trajectory server")

    updateCurrentState()
    # From the EE to POI
    T_W_poi_pos, T_W_poi_quat = getSiteTransformWRTWorld('poi')
    T_W_ee_pos, T_W_ee_quat = getSiteTransformWRTWorld('ee')
    T_ee_poi_pos, T_ee_poi_quat = findLocalTransform(T_W_ee_pos, T_W_ee_quat, T_W_poi_pos, T_W_poi_quat)
    T_ee_poi_rot = np.zeros(9)
    mp.mju_quat2Mat(T_ee_poi_rot, T_ee_poi_quat)

    print('T_ee_poi_pos: ', T_ee_poi_pos)
    print('T_ee_poi_rot', T_ee_poi_rot)
    T_ee_poi = np.concatenate([T_ee_poi_pos, T_ee_poi_rot])
    np.savetxt(os.path.join(save_dir, 'T_ee_poi.txt'), T_ee_poi.reshape((1,12)))

    # From the base to camera frame (the camera extrinsics)
    T_W_base_pos, T_W_base_quat = getSiteTransformWRTWorld('base')
    T_W_cam1_pos, T_W_cam1_quat = getSiteTransformWRTWorld('cam1')
    T_base_cam1_pos, T_base_cam1_quat = findLocalTransform(T_W_base_pos, T_W_base_quat, T_W_cam1_pos, T_W_cam1_quat)
    T_base_cam1_rot = np.zeros(9)
    mp.mju_quat2Mat(T_base_cam1_rot, T_base_cam1_quat)

    print('T_base_cam1_pos: ', T_base_cam1_pos)
    print('T_base_cam1_rot', T_base_cam1_rot)

    # get the virtual bounding box to pick calibration targets
    calib_target = getCalibTarget(model, data)

    # go to start calib pose
    if isValidTrajectory(start_tpva):
        if sim:
            visualize(start_tpva)
        else:
            executeTraj(start_tpva)
    else:
        exit(0)

    count = 0
    fail = 0
    calib_data = np.empty((num_samples, 13))
    while count < num_samples:
        print("==========================================================")
        print("================ Sample No:", count+1, "==================")
        print("==========================================================")
        s = sampleFromCalibTarget(calib_target)
        ik_result = qpos_from_site_pose(phy, 'calibtool_center', s[0], s[1], manip_joints, inplace=False)
        if ik_result.success:
            fail = 0
            updateCurrentState()
            # print("Generating trajectory from ", curr_state.actual.positions, " to ", ik_result.qpos)
            print("Successfully generated collision-free trajectory. Now executing!!!")
            tpva = generateTrajectory(curr_state.actual.positions, ik_result.qpos)
            if isValidTrajectory(tpva):
                if sim:
                    visualize(tpva)
                else:
                    executeTraj(tpva)
                    clickAndSavePic(count+1)

                    T_W_ee_pos, T_W_ee_quat = getSiteTransformWRTWorld('ee')
                    T_W_base_pos, T_W_base_quat = getSiteTransformWRTWorld('base')
                    T_base_ee_pos, T_base_ee_quat = findLocalTransform(T_W_base_pos, T_W_base_quat, T_W_ee_pos, T_W_ee_quat)
                    T_base_ee_rot = np.zeros(9)
                    mp.mju_quat2Mat(T_base_ee_rot, T_base_ee_quat)

                    calib_data[count][:] = np.concatenate([np.array([count+1], dtype=int), T_base_ee_pos, T_base_ee_rot]).T

                count = count + 1

            else:
                print('Sampling new pose to avoid collision')
        else:
            fail = fail + 1
            print("ik failed to ", s[0], s[1])
            if fail > 10:
                print("IK failed 10 consecutive times. Exiting")
                exit(0)
                break

    tpva = generateTrajectory(curr_state.actual.positions, calib_end_start)
    if isValidTrajectory(tpva):
        if sim:
            visualize(tpva)
        else:
            executeTraj(tpva)

    if isValidTrajectory(end_tpva):
        if sim:
            visualize(end_tpva)
        else:
            executeTraj(end_tpva)

    if not sim:
        np.savetxt(os.path.join(save_dir, 'T_base_ee.txt'), calib_data)

