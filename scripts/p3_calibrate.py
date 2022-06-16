import numpy as np
import pickle
import glob
from scipy.optimize import fmin_powell
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo
# from sklearn.decomposition import PCA

# from skspatial.objects import Plane
# from skspatial.objects import Points
# from skspatial.plotting import plot_3d

# from numba.typed import List
# from numba import njit

## Global config variables
mass_ball = 0.03135
gravity = -9.80665
g_A_FP_SC = np.array([[ 0.06504259, -0.08839325,  0.99395981,  6.01944448],
       [ 0.99754884, -0.01999624, -0.06705572, -0.52128749],
       [ 0.02580274,  0.99588492,  0.08687598,  0.55443587],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
radius_ball = 0.0381

## Global runtime variables
g_sc_points = []
g_sc_times = []
g_ki_points = []
g_ki_times = []

def extrapolate_projectile_to_time(px,py,pz,vx,vy,vz,delt,C):
    global gravity, mass_ball
    g = -gravity
    vt = -mass_ball*gravity/C
    ez = pz + (vt/g)*(vz+vt)*(1.0-np.exp(-(g*delt/vt)))-vt*delt
    ex = px + (vx*vt/g)*(1.0-np.exp(-g*delt/vt))
    ey = py + (vy*vt/g)*(1.0-np.exp(-g*delt/vt))
    return np.array([ex,ey,ez])

def compute_single_error(sc_points, sc_times, ki_points, ki_times, params):
    global g_A_FP_SC, gravity, mass_ball
    ## Convert SC points to PR2 frame
    x_off, y_off, z_off, roll, pitch, yaw, C = params
    res_pitch_A = np.array([[1,0,0,0],
                      [0,np.cos(pitch),-np.sin(pitch),0],
                      [0,np.sin(pitch),np.cos(pitch),0],
                      [0,0,0,1]])
    res_yaw_A = np.array([[np.cos(yaw),0,np.sin(yaw),0],
                      [0,1.0,0,0],
                      [-np.sin(yaw),0,np.cos(yaw),0],
                      [0,0,0,1]])
    res_roll_A = np.array([[np.cos(roll),-np.sin(roll),0,0],
                      [np.sin(roll),np.cos(roll),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    pr2_frame_points = np.linalg.multi_dot([g_A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
    pr2_frame_points += np.array([[x_off],[y_off],[z_off],[1.0]])

    ## Compute projectile velocity
    s, e = 0, pr2_frame_points.shape[1]-1
    vxm = []
    vym = []
    vzim = []

    vt = -mass_ball*gravity/C

    for i in range(1,e+1):
        delx = pr2_frame_points[0,i] - pr2_frame_points[0,s]
        dely = pr2_frame_points[1,i] - pr2_frame_points[1,s]
        delz = pr2_frame_points[2,i] - pr2_frame_points[2,s]
        delt = sc_times[i] - sc_times[s]
        # vx = delx/delt
        # vy = dely/delt
        # vzi = (delz-0.5*gravity*delt*delt)/delt
        vx = (delx*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        vy = (dely*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        vzi = (((delz+vt*delt)*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))) - vt
        vxm.append(vx)
        vym.append(vy)
        vzim.append(vzi)
    vx = np.mean(vxm)
    vy = np.mean(vym)
    vzi = np.mean(vzim)

    extrapolated_points = np.array([extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],vx,vy,vzi,delt,C ) for delt in ki_times])
    error = np.mean(np.power(extrapolated_points.transpose()-ki_points,2))
    return error

def correct_pc_points(pc_points):
    # return pc_points
    direction = pc_points[0:3,:]/np.linalg.norm(pc_points[0:3,:],axis=0)
    correct_pc_points = copy.deepcopy(pc_points)
    correct_pc_points[0:3,:] = correct_pc_points[0:3,:] + direction*0.0381
    return correct_pc_points

def compute_parabola_error(sc_points, sc_times, ki_points, ki_times, params, wall_x=1.0):
    global g_A_FP_SC, gravity, mass_ball
    ## Convert SC points to PR2 frame
    x_off, y_off, z_off, roll, pitch, yaw, C = params
    res_pitch_A = np.array([[1,0,0,0],
                      [0,np.cos(pitch),-np.sin(pitch),0],
                      [0,np.sin(pitch),np.cos(pitch),0],
                      [0,0,0,1]])
    res_yaw_A = np.array([[np.cos(yaw),0,np.sin(yaw),0],
                      [0,1.0,0,0],
                      [-np.sin(yaw),0,np.cos(yaw),0],
                      [0,0,0,1]])
    res_roll_A = np.array([[np.cos(roll),-np.sin(roll),0,0],
                      [np.sin(roll),np.cos(roll),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    pr2_frame_points = np.linalg.multi_dot([g_A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
    pr2_frame_points += np.array([[x_off],[y_off],[z_off],[1.0]])

    vt = -mass_ball*gravity/C
    g = -gravity
    ## Compute projectile velocity
    s, e = 0, pr2_frame_points.shape[1]-1
    vxm = []
    vym = []
    vzim = []
    for i in range(1,e+1):
        delx = pr2_frame_points[0,i] - pr2_frame_points[0,s]
        dely = pr2_frame_points[1,i] - pr2_frame_points[1,s]
        delz = pr2_frame_points[2,i] - pr2_frame_points[2,s]
        delt = sc_times[i] - sc_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        vxm.append(vx)
        vym.append(vy)
        vzim.append(vzi)
    sc_vx = np.mean(vxm)
    sc_vy = np.mean(vym)
    sc_vzi = np.mean(vzim)

    ## Compute velocity of GT projectile
    s, e = 0, ki_points.shape[1]-1
    k_vxm = []
    k_vym = []
    k_vzim = []
    for i in range(1,e+1):
        delx = ki_points[0,i] - ki_points[0,s]
        dely = ki_points[1,i] - ki_points[1,s]
        delz = ki_points[2,i] - ki_points[2,s]
        delt = ki_times[i] - ki_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        k_vxm.append(vx)
        k_vym.append(vy)
        k_vzim.append(vzi)
    k_vx = np.mean(k_vxm)
    k_vy = np.mean(k_vym)
    k_vzi = np.mean(k_vzim)
    
    ki_delx = wall_x-ki_points[0,0]
    ki_delt = ki_delx/(k_vx)
    ki_point = extrapolate_projectile_to_time( ki_points[0,0],ki_points[1,0],ki_points[2,0],k_vx,k_vy,k_vzi,ki_delt,C)
    reference_points = np.concatenate([ki_points, ki_point.reshape(-1,1)], axis=1)

    extrapolated_points = np.array([extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],sc_vx,sc_vy,sc_vzi,delt,C ) for delt in np.arange(ki_times[0]-0.030, ki_delt+ki_times[0], 0.005)])
    extrapolated_points = extrapolated_points.transpose()

    diff_list = []
    for i in range(reference_points.shape[1]):
        diff = np.min(np.linalg.norm(extrapolated_points-reference_points[:,i:i+1], axis=0))
        diff_list.append(diff)

    error = np.mean(diff_list)
    return error

def compute_single_error_wall_plane(sc_points, sc_times, ki_points, ki_times, params, wall_x=1.0):
    global g_A_FP_SC, gravity, mass_ball
    ## Convert SC points to PR2 frame
    x_off, y_off, z_off, roll, pitch, yaw, C = params
    res_pitch_A = np.array([[1,0,0,0],
                      [0,np.cos(pitch),-np.sin(pitch),0],
                      [0,np.sin(pitch),np.cos(pitch),0],
                      [0,0,0,1]])
    res_yaw_A = np.array([[np.cos(yaw),0,np.sin(yaw),0],
                      [0,1.0,0,0],
                      [-np.sin(yaw),0,np.cos(yaw),0],
                      [0,0,0,1]])
    res_roll_A = np.array([[np.cos(roll),-np.sin(roll),0,0],
                      [np.sin(roll),np.cos(roll),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    pr2_frame_points = np.linalg.multi_dot([g_A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
    pr2_frame_points += np.array([[x_off],[y_off],[z_off],[1.0]])

    vt = -mass_ball*gravity/C
    g = -gravity
    ## Compute projectile velocity
    s, e = 0, pr2_frame_points.shape[1]-1
    vxm = []
    vym = []
    vzim = []
    for i in range(1,e+1):
        delx = pr2_frame_points[0,i] - pr2_frame_points[0,s]
        dely = pr2_frame_points[1,i] - pr2_frame_points[1,s]
        delz = pr2_frame_points[2,i] - pr2_frame_points[2,s]
        delt = sc_times[i] - sc_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        # vx = (delx*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        # vy = (dely*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        # vzi = (((delz+vt*delt)*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))) - vt
        vxm.append(vx)
        vym.append(vy)
        vzim.append(vzi)
    sc_vx = np.mean(vxm)
    sc_vy = np.mean(vym)
    sc_vzi = np.mean(vzim)
    # print(sc_vx, np.std(vxm), " | ", sc_vy, np.std(vym), " | ", sc_vzi, np.std(vzim), " SC ")

    ## Compute velocity of GT projectile
    s, e = 0, ki_points.shape[1]-1
    k_vxm = []
    k_vym = []
    k_vzim = []
    for i in range(1,e+1):
        delx = ki_points[0,i] - ki_points[0,s]
        dely = ki_points[1,i] - ki_points[1,s]
        delz = ki_points[2,i] - ki_points[2,s]
        delt = ki_times[i] - ki_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        # vx = (delx*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        # vy = (dely*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))
        # vzi = (((delz+vt*delt)*gravity)/(vt*(1.0-np.exp(-gravity*delt/vt)))) - vt
        k_vxm.append(vx)
        k_vym.append(vy)
        k_vzim.append(vzi)
    k_vx = np.mean(k_vxm)
    k_vy = np.mean(k_vym)
    k_vzi = np.mean(k_vzim)
    # print(k_vx, np.std(k_vxm), " | ", k_vy, np.std(k_vym), " | ", k_vzi, np.std(k_vzim), " KI ")
    
    ## Compute Errors
    diff_list = []

    ## Error for the middle points
    extrapolated_points = np.array([extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],sc_vx,sc_vy,sc_vzi,delt,C ) for delt in ki_times])
    diff_list.extend( (extrapolated_points.transpose()-ki_points).flatten().tolist() )

    ## Error for the wall point
    ki_delx = wall_x-ki_points[0,0]
    # ki_delt = -vt/g*np.log(1.0-(ki_delx*g/(k_vx*vt)))
    # print(ki_delt)
    # ki_delt = 0.06 if ki_delt>0.08 or ki_delt<0.01 else ki_delt
    ki_delt = ki_delx/(k_vx)
    # ki_delt = 0.11
    ki_point = extrapolate_projectile_to_time( ki_points[0,0],ki_points[1,0],ki_points[2,0],k_vx,k_vy,k_vzi,ki_delt,C )
    sc_point = extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],sc_vx,sc_vy,sc_vzi,ki_delt+ki_times[0],C )
    diff_list.extend( (ki_point-sc_point).flatten().tolist() )

    # error = np.mean(np.power(diff_list,2))
    error = np.mean(np.abs(diff_list))

    return error

def visualize_wall_plane(sc_points, sc_times, ki_points, ki_times, params, wall_x=1.0):
    global g_A_FP_SC, gravity, mass_ball
    ## Convert SC points to PR2 frame
    x_off, y_off, z_off, roll, pitch, yaw, C = params
    res_pitch_A = np.array([[1,0,0,0],
                      [0,np.cos(pitch),-np.sin(pitch),0],
                      [0,np.sin(pitch),np.cos(pitch),0],
                      [0,0,0,1]])
    res_yaw_A = np.array([[np.cos(yaw),0,np.sin(yaw),0],
                      [0,1.0,0,0],
                      [-np.sin(yaw),0,np.cos(yaw),0],
                      [0,0,0,1]])
    res_roll_A = np.array([[np.cos(roll),-np.sin(roll),0,0],
                      [np.sin(roll),np.cos(roll),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    pr2_frame_points = np.linalg.multi_dot([g_A_FP_SC,res_roll_A,res_pitch_A,res_yaw_A,sc_points])
    pr2_frame_points += np.array([[x_off],[y_off],[z_off],[1.0]])

    vt = -mass_ball*gravity/C
    g = -gravity
    ## Compute projectile velocity
    s, e = 0, pr2_frame_points.shape[1]-1
    vxm = []
    vym = []
    vzim = []
    for i in range(1,e+1):
        delx = pr2_frame_points[0,i] - pr2_frame_points[0,s]
        dely = pr2_frame_points[1,i] - pr2_frame_points[1,s]
        delz = pr2_frame_points[2,i] - pr2_frame_points[2,s]
        delt = sc_times[i] - sc_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        vxm.append(vx)
        vym.append(vy)
        vzim.append(vzi)
    sc_vx = np.mean(vxm)
    sc_vy = np.mean(vym)
    sc_vzi = np.mean(vzim)

    ## Compute velocity of GT projectile
    s, e = 0, ki_points.shape[1]-1
    k_vxm = []
    k_vym = []
    k_vzim = []
    for i in range(1,e+1):
        delx = ki_points[0,i] - ki_points[0,s]
        dely = ki_points[1,i] - ki_points[1,s]
        delz = ki_points[2,i] - ki_points[2,s]
        delt = ki_times[i] - ki_times[s]
        vx = delx/delt
        vy = dely/delt
        vzi = (delz-0.5*gravity*delt*delt)/delt
        k_vxm.append(vx)
        k_vym.append(vy)
        k_vzim.append(vzi)
    k_vx = np.mean(k_vxm)
    k_vy = np.mean(k_vym)
    k_vzi = np.mean(k_vzim)
    
    extrapolated_points = np.array([extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],sc_vx,sc_vy,sc_vzi,delt,C ) for delt in ki_times])

    ## for the wall point
    ki_delx = wall_x-ki_points[0,0]
    ki_delt = ki_delx/(k_vx)
    ki_point = extrapolate_projectile_to_time( ki_points[0,0],ki_points[1,0],ki_points[2,0],k_vx,k_vy,k_vzi,ki_delt,C )
    sc_point = extrapolate_projectile_to_time( pr2_frame_points[0,0],pr2_frame_points[1,0],pr2_frame_points[2,0],sc_vx,sc_vy,sc_vzi,ki_delt+ki_times[0],C )

    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(pr2_frame_points[0,:], pr2_frame_points[1,:], pr2_frame_points[2,:], color = "green")
    ax.scatter3D(ki_points[0,:], ki_points[1,:], ki_points[2,:], color = "red")
    ax.scatter3D(extrapolated_points[:,0], extrapolated_points[:,1], extrapolated_points[:,2], color = "black")
    ax.scatter3D(ki_point[0], ki_point[1], ki_point[2], color = "blue")
    ax.scatter3D(sc_point[0], sc_point[1], sc_point[2], color = "green")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.show()
    return 0

def compute_multiple_error(params):
    #params = (0,0,0,0,0,0,params[6])
    global g_sc_points, g_sc_times, g_ki_points, g_ki_times
    mean_error = [compute_parabola_error(sc_points,sc_times,ki_points,ki_times,params,wall_x=0.0) for (sc_points,sc_times,ki_points,ki_times) in tuple(zip(g_sc_points,g_sc_times,g_ki_points,g_ki_times)) ]
    return np.mean(mean_error)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def spherePoints(cx,cy,cz,r):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*r + cx
    y = np.sin(u)*np.sin(v)*r + cy
    z = np.cos(v)*r + cz
    return x,y,z

# @njit(nopython=True)
def convert2id(mins, point):
    v = (point-mins)*1000
    return int(v[0]), int(v[1]), int(v[2])

# @njit(nopython=True)
def convert2location(mins, point):
    return (point/1000.0) + mins

# @njit(nopython=True)
def find_center(ball_points, threshold):
    # mins = np.array([np.min(ball_points[0]), np.min(ball_points[1]), np.min(ball_points[2])])-radius_ball-0.01
    mins = np.min(ball_points[0], axis=0)-radius_ball-0.01
    # maxs = np.array([np.max(ball_points[0]), np.max(ball_points[1]), np.max(ball_points[2])])+radius_ball+0.01
    maxs = np.max(ball_points, axis=0)+radius_ball+0.01
    vol_size = convert2id(mins, maxs)
    vote = np.zeros(vol_size, dtype=np.uint16)
    max_vote = 0
    delangle = 5*np.pi/180
    for i in range(ball_points.shape[0]):
        for theta in np.arange(0*np.pi/2,4*np.pi/2,delangle):
            for phi in np.arange(0,np.pi,delangle):
                cx = ball_points[i,0]-radius_ball*np.cos(theta)*np.sin(phi)
                cy = ball_points[i,1]-radius_ball*np.sin(theta)*np.sin(phi)
                cz = ball_points[i,2]-radius_ball*np.cos(phi)
                cnp = np.array([cx,cy,cz])
                bv = cnp-ball_points[i,:]
                if np.dot(bv,ball_points[i,:]) > 0:
                    ids = convert2id(mins,np.array([cx,cy,cz]))
                    vote[ids[0], ids[1], ids[2]] += 1
                    if vote[ids[0], ids[1], ids[2]] > max_vote:
                        max_vote = vote[ids[0], ids[1], ids[2]]
    # print("Done voting")
    other_centers = []
    center_votes = []
    locs = np.argwhere(vote >= max_vote*threshold)
    for idx in range(locs.shape[0]):
        vote_pos = locs[idx,:]
        ch = convert2location(mins, vote_pos)
        other_centers.append(ch)
        center_votes.append(vote[vote_pos[0], vote_pos[1], vote_pos[2]]/ball_points.shape[0])
    # center = np.sum(other_centers*np.array(center_votes).reshape(-1,1),axis=0)/np.sum(center_votes)
    # center = np.mean(other_centers, axis=0)
    # print("Finding mean")
    return other_centers, center_votes

# def find_radius_vote(pointcloud_list):
#     estimates = []
#     pca_scores = []
#     found = 0
#     # fig = plt.figure(figsize = (10, 10))
#     # ax = plt.axes(projection ="3d")
#     for ball_points in pointcloud_list:
#         pca = PCA(n_components=3)
#         pca.fit(ball_points)
#         print("PCA", pca.singular_values_[0]/pca.singular_values_[1], pca.singular_values_[1]/pca.singular_values_[2], pca.components_)
#         pca_score = pca.singular_values_[0]/pca.singular_values_[1]
            
#         other_centers, center_votes = find_center(ball_points, 1)
#         center = np.mean(np.array(other_centers), axis=0)
#         estimates.append(center)
#         pca_scores.append(pca_score)


#     #     # Creating plot
#     #     ax.scatter3D(ball_points[:,0], ball_points[:,1], ball_points[:,2], color = "green")
#     #     # ax.scatter3D(other_centers[:,0], other_centers[:,1], other_centers[:,2], color = "red")
#     #     #ax.scatter3D(0, 0, 0, color = "blue")
#     #     # ax.quiver(0,0,0,ball_position_estimate[0,0], ball_position_estimate[1,0],ball_position_estimate[2,0],length=0.1)
#     #     ax.plot([center[0]*0.9,center[0]],[center[1]*0.9,center[1]],[center[2]*0.9,center[2]],'-')
#     #     # sx,sy,sz = spherePoints(ball_position_estimate[0,0], ball_position_estimate[1,0],ball_position_estimate[2,0],radius_ball)
#     #     # ax.plot_wireframe(sx, sy, sz, color="b")
#     #     sx,sy,sz = spherePoints(center[0], center[1],center[2],radius_ball)
#     #     ax.plot_wireframe(sx, sy, sz, color="yellow")
    
#     # set_axes_equal(ax)
#     # plt.show()
#     return estimates, pca_scores

# def compute_fit_score(center, ball_points):
#     return np.mean(np.abs(np.linalg.norm(ball_points-center.reshape(1,-1))-radius_ball))


# def find_radius_traj_vote(pointcloud_list):
#     ball_centers = []
#     coords = []
#     pca_scores = []
#     estimates = []
#     for ball_points in pointcloud_list:
#         pca = PCA(n_components=3)
#         pca.fit(ball_points)
#         print("PCA", pca.singular_values_[0]/pca.singular_values_[1], pca.singular_values_[1]/pca.singular_values_[2], pca.components_)
#         pca_score = pca.singular_values_[0]/pca.singular_values_[1]    
#         other_centers, center_votes = find_center(ball_points, 1.0)
#         ball_centers.append(other_centers)
#         coords.extend(other_centers)
#         center = np.sum(other_centers*np.array(center_votes).reshape(-1,1),axis=0)/np.sum(center_votes)
#         #print(np.mean(center_votes), "votes", np.max(center_votes), compute_fit_score(center, ball_points), "\n")
#         estimates.append(center)
#         pca_scores.append(compute_fit_score(center, ball_points))
#     # points = Points(coords)
#     # plane = Plane.best_fit(points)
#     # for i in range(len(ball_centers)):
#     #     best_center = None
#     #     min_distance = np.inf
#     #     for j in range(len(ball_centers[i])):
#     #         distance = plane.distance_point(ball_centers[i][j])
#     #         if distance < min_distance:
#     #             min_distance = distance
#     #             best_center = ball_centers[i][j]
#     #     estimates.append(best_center)
#     # estimates = np.array(estimates)
    
#     # center_points = Points(estimates)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # plot_3d(
#     #     # points.plotter(c='k', s=50, depthshade=False),
#     #     plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
#     #     center_points.plotter(c='r', s=50, depthshade=False),
#     # )
#     # plt.show()


#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     # plot_3d(
#     #     points.plotter(c='k', s=50, depthshade=False),
#     #     plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
#     # )
#     # plt.show()

#     # fig = plt.figure(figsize = (10, 10))
#     # ax = plt.axes(projection ="3d")
#     # ball_Ppoints = []
#     # for i, ball_points in enumerate(pointcloud_list):
#     #     center = estimates[i]
#     #     ax.scatter3D(ball_points[:,0], ball_points[:,1], ball_points[:,2], color = "green")
#     #     ax.plot([center[0]*0.9,center[0]],[center[1]*0.9,center[1]],[center[2]*0.9,center[2]],'-')
#     #     sx,sy,sz = spherePoints(center[0], center[1],center[2],radius_ball)
#     #     ax.plot_wireframe(sx, sy, sz, color="yellow")
#     #     ball_Ppoints.extend(ball_points.tolist())
#     # ball_Ppoints = Points(ball_Ppoints)    
#     # set_axes_equal(ax)
#     # plt.show()
#     return estimates, pca_scores



def main():
    global g_sc_points, g_sc_times, g_ki_points, g_ki_times

    param_file_url = "../data_dump/perception_params.npy"
    data_files = glob.glob("../data_dump/calib_data_runs/*.pickle")
    # data_files = glob.glob("/home/roman/catkin_ws/calib_data/*.pickle")
    selected_files = data_files
    for file_name in selected_files:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle, encoding='latin1')
            # sc_pc_points,pca_scores = find_radius_traj_vote(data['sc_points'])
            # data['sc_centers'] = sc_pc_points
            # data['pca_scores'] = pca_scores
            # with open(file_name+'mod2.pkl', 'wb') as fp:
            #     pickle.dump(data, fp)

            # sc_pc_points  = data['sc_centers']
            # pca_scores = data['pca_scores']
            sc_times = np.array(data['sc'][1])-data['sc'][1][0]
            ki_times = np.array(data['ki'][1])-data['sc'][1][0]
            ki_points = np.array(data['ki'][0]).transpose()

            sc_points = data['sc'][0]
            sc_points = np.concatenate([np.array(sc_points).transpose(), np.ones((1,len(sc_points)))], axis=0)
            g_sc_points.append(sc_points)
            g_sc_times.append(sc_times)
            g_ki_points.append(ki_points)
            g_ki_times.append(ki_times)

            if(sc_points.shape[1] != 5 or ki_points.shape[1] != 3 or ki_times.shape[0] != 3 or sc_times.shape[0] != 5):
                print("Bigg error", file_name, sc_points.shape)

            # f_sc_points = []
            # f_sc_times = []
            # for i in range(len(pca_scores)):
            #     if pca_scores[i] <= 1.5:
            #         f_sc_points.append(sc_pc_points[i])
            #         f_sc_times.append(sc_times[i])
            # f_sc_points = np.array(f_sc_points)
            # print(f_sc_points.shape[0], "Shapes")
            # if(f_sc_points.shape[0] >= 3):
            #     f_sc_points = np.concatenate([np.array(f_sc_points).transpose(), np.ones((1,len(f_sc_points)))], axis=0)

            #     g_sc_points.append(f_sc_points)
            #     g_sc_times.append(f_sc_times)
            #     g_ki_points.append(ki_points)
            #     g_ki_times.append(ki_times)
    
    # params = list((0,0,0,0,0,0,np.random.uniform(0.01, 0.03)))
    # params = list(np.load(param_file_url).tolist())
    params = [0.404571, 0.13428446, 0.35198332, 1.5798978885211274, -8.995515878771856, -0.34771949149796966, 0.01176908]
    params[3] = params[3]*np.pi/180.0
    params[4] = params[4]*np.pi/180.0
    params[5] = params[5]*np.pi/180.0
    print("Initial params", params)
    print("Cost of initial params",compute_multiple_error(params),np.sqrt(compute_multiple_error(params)))
    best_params = fmin_powell(compute_multiple_error, params, ftol=0.00001, xtol=0.00001)
    print("Cost of best params",compute_multiple_error(best_params), np.sqrt(compute_multiple_error(best_params)))
    best_params[3] = best_params[3]*180.0/np.pi
    best_params[4] = best_params[4]*180.0/np.pi
    best_params[5] = best_params[5]*180.0/np.pi
    np.save(param_file_url,np.array(best_params))
    print("Best params", best_params)

main()
