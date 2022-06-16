import numpy as np
import pickle

f = open("tf_data.pkl",'rb')
data = pickle.load(f, fix_imports=True, encoding="latin1")
f.close()

list_tf_1 = data[1]
list_tf_2 = data[0]

list_p1 = []
list_p2 = []

for i in range(len(list_tf_1)):
    if i == 13:
        continue
    list_p1.append(list_tf_1[i][0])
    list_p2.append(list_tf_2[i][0])

p1_t = np.array(list_p1)
p2_t = np.array(list_p2)

p1 = p1_t.transpose()
p2 = p2_t.transpose()

p1_c = np.mean(p1,axis=1).reshape((-1,1))
p2_c = np.mean(p2,axis=1).reshape((-1,1))

q1 = p1-p1_c
q2 = p2-p2_c

H=np.matmul(q1,q2.transpose())

U,X,V_t = np.linalg.svd(H)

R = np.matmul(V_t.transpose(),U.transpose())

assert np.allclose(np.linalg.det(R),1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al"

# Calculate translation matrix
T = p2_c - np.matmul(R,p1_c)

print ("translation", T, "\n\n Rotation \n", R)

result = T+np.matmul(R,p1)
if np.allclose(result,p2):
    print("transformation is correct!")
else:
    print("transformation is wrong ..")
    print(np.mean(result-p2))


from scipy.spatial.transform import Rotation as sR 
r = sR.from_dcm(R)
print(T.transpose(), '\n\n', r.as_quat())

## 0.5203829, 0.4456246, 0.4850521, 0.5434564