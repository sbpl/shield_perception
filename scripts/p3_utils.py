from scipy.spatial.transform import Rotation as R
import numpy as np


m_FR_KI = np.array([[5.6728428586518398e-01, -1.8312947703924448e-01,
       8.0290231887182950e-01, 2.2096061713157075e-02,
       9.7799527024867916e-01, 2.0745364645621767e-01,
       -8.2322554811375748e-01, -9.9944214492614938e-02,
       5.5884778868870699e-01]]).reshape(3,3)

v_FR_KI = np.array([[-3.8316579018879668e+00], [-2.5176660993665356e-01], [-5.5160582917909567e-02]])

r_FR_KI = R.from_matrix(m_FR_KI)

R_KI_FR = r_FR_KI.inv().as_matrix()
T_KI_FR = -R_KI_FR@v_FR_KI

r_KI_FR = R.from_matrix(R_KI_FR)

q_KI_FR = r_KI_FR.as_quat() #(x, y, z, w) format
print("Quaternion KI_FR", q_KI_FR) 
print("Translation KI_FR", T_KI_FR) 

#### Fridge
m_RS_FR = np.array([[  -8.0175717891370379e-01, 1.5653112260307932e-02,
       -5.9744489799219591e-01, -9.9288028845320314e-03,
       -9.9986785125583499e-01, -1.2872408411776694e-02,
       -5.9756743965314840e-01, -4.3886332275172519e-03,
       8.0180664437554694e-01]]).reshape(3,3)

v_RS_FR = np.array([[3.6023059399139927e+00, 2.7793609352959769e-01,
    7.5353336685286787e-01]]).T

r_RS_FR = R.from_matrix(m_RS_FR)

R_FR_RS = r_RS_FR.inv().as_matrix()
T_FR_RS = -R_FR_RS@v_RS_FR

r_FR_RS = R.from_matrix(R_FR_RS)

q_FR_RS = r_FR_RS.as_quat() #(x, y, z, w) format
print("Quaternion FR_RS", q_FR_RS) 
print("Translation FR_RS", T_FR_RS) 


### Common
m_SC_RS = np.array([[ -3.1564279814339152e-01, 9.4861512326541497e-01,
       -2.2337678759259905e-02, -1.2114171444410449e-01,
       -1.6937924425402318e-02, 9.9249070108374449e-01,
       9.4111333483367110e-01, 3.1597856672296804e-01,
       1.2026319624035342e-01]]).reshape(3,3)

v_SC_RS = np.array([[3.0507754950657190e-01, -2.1872877885724251e+00,
    3.1787317506636570e+00]]).T

r_SC_RS = R.from_matrix(m_SC_RS)

R_RS_SC = r_SC_RS.inv().as_matrix()
T_RS_SC = -R_RS_SC@v_SC_RS

r_RS_SC = R.from_matrix(R_RS_SC)

q_RS_SC = r_RS_SC.as_quat() #(x, y, z, w) format
print("Quaternion RS_SC", q_RS_SC) 
print("Translation RS_SC", T_RS_SC) 

v_KI_FP = np.array([[-0.0897972122212618, 0.008121730656646616, 1.5861054037696538]]).T
R_FP_KI = R.from_quat([-0.5365288218513303, 0.5513395064561633, -0.4578609339676863, 0.44556137297765813]).inv().as_matrix()
T_FP_KI = -R_FP_KI@v_KI_FP



### Setting the transformation matrices
A_FP_KI = np.eye(4)
A_FP_KI[0:3,0:3] = R_FP_KI
A_FP_KI[0:3,3:4] = T_FP_KI

A_KI_FR = np.eye(4)
A_KI_FR[0:3,0:3] = R_KI_FR
A_KI_FR[0:3,3:4] = T_KI_FR

A_FR_RS = np.eye(4)
A_FR_RS[0:3,0:3] = R_FR_RS
A_FR_RS[0:3,3:4] = T_FR_RS

A_RS_SC = np.eye(4)
A_RS_SC[0:3,0:3] = R_RS_SC
A_RS_SC[0:3,3:4] = T_RS_SC

# A_FP_SC = A_FP_KI @ A_KI_FR @ A_FR_RS @ A_RS_SC





A_FP_SC = np.eye(4)
r_FP_SC = R.from_quat([0.44077743511951256, 0.43098939172865447, 0.5345096186509952, 0.5781547013234453])
v_FP_SC = np.array([5.574363423212863, -0.40031510516119584, 0.577317012401472])
R_FP_SC = r_FP_SC.as_matrix()
A_FP_SC[0:3,0:3] = R_FP_SC
A_FP_SC[0:3,3] = v_FP_SC

p = np.array([[0],[0],[2],[1]])
p_PR = A_FP_SC@p
print(p_PR)

np.save("/home/roman/catkin_ws/src/shield_planner/shield_perception/A_FP_SC.npy", A_FP_SC)



KI_pos = np.array([ 8.75553005846, -1.44091180176,  0.        ])
KI_vel = np.array([-6.46536633851,  2.30268905834,  5.94069514121])
SC_pos = np.array([ 10.0067197637, -1.57587298998,  0.        ])
SC_vel = np.array([-8.32213973999, 2.26484370232, 6.06468248367])


KI_vel_dir = KI_vel/np.linalg.norm(KI_vel)
SC_vel_dir = SC_vel/np.linalg.norm(SC_vel)


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


R_SC_KI = ComputeRotationMSO(SC_vel_dir, KI_vel_dir)
KI2_vel = R_SC_KI @ SC_vel.reshape((-1,1))

print("KI2 VEL ", KI2_vel)
print("KI VEL ", KI_vel)

T_SC_KI = KI_pos.reshape((-1,1)) - R_SC_KI @ SC_pos.reshape((-1,1))

print("T_SC_KI", T_SC_KI)

print("After Calibration")
print("Position:", R_SC_KI @ SC_pos.reshape((-1,1)) + T_SC_KI)
print("Veclocity:", R_SC_KI @ SC_vel.reshape((-1,1)))

print(KI_pos)

save_dir = "/home/roman/catkin_ws/src/shield_planner/shield_perception/"
np.save(save_dir+"/R_SC_KI.npy", R_SC_KI)
np.save(save_dir+"/T_SC_KI.npy", T_SC_KI)



#### SC to Kinnect
A_SC_KI = np.array([ -0.998984, -0.034984, 0.028420, 0.288399,
0.030982, -0.990931, -0.130749, 1.601808,
0.032737, -0.129736, 0.991008, 5.616120,
0.000000, 0.000000, 0.000000, 1.000000]).reshape(4,4)

m_SC_KI = A_SC_KI[0:3,0:3]

v_SC_KI = A_SC_KI[0:3,3:4]

q_SC_KI = R.from_matrix(m_SC_KI).as_quat()

print("Quaternion SC_KI", q_SC_KI) 
print("Translation SC_KI", v_SC_KI[:,0]) 