import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pdb

data_dir = "../data_dump/calib_data_runs/"
bad_data_dir = os.path.join(data_dir, "bad")

if not os.path.isdir(bad_data_dir):
    os.mkdir(bad_data_dir)

all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path. join(data_dir, f))]

fig = plt.figure()
ax = plt.axes(projection='3d')
i = 0

print("Total files: {}".format(len(all_files)))

for file in all_files:
    
    if file == ".dummy":
        continue
    
    i+=1
    print("file {}: {}".format(i, file))

    file = os.path.join(data_dir, file)
    
    ax.clear()

    with open(file, 'rb') as handle:
        data = pickle.load(handle, encoding='latin1')

    for i in range(5):
        ax.plot3D(data['sc_points'][i][:,0], data['sc_points'][i][:,1], data['sc_points'][i][:,2])

    for i in range(3):
        ax.plot3D(data['ki_points'][i][:,0], data['ki_points'][i][:,1], data['ki_points'][i][:,2])
    
    plt.show(block=False)    

    delt_t  = []

    for i in range(5):
        delt = data['sc'][1][i]-data['sc'][1][0]
        delt_t.append(delt)

    for i in range(3):
        delt = data['ki'][1][i]-data['sc'][1][0]
        delt_t.append(delt)

    print("Time stamps delta: {}".format(delt_t))

    value = input("Enter b for bad data, else just hit enter ")
    print("Entered key: ".format(value))

    if value == 'b':
        os.popen("mv " + file + " " + bad_data_dir)


    handle.close()

    

