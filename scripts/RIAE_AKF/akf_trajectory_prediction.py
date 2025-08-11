from CoordsKF import CoordsKF
import visualize_mean_cov
import numpy as np
import json
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'groundTruthMeasurements.json')

with open(file_path, "r") as f:
    mean_list=[]
    cov_list=[]
    measurements = []
    gt_time_coords = json.load(f)
    
    i = 0
    for row in gt_time_coords.values():
        # For one throw
        if i == 2:  
            print("i", i)
            row = np.array(row)
            measurements = row[:,0:4]

            dt0 = measurements[1,0] - measurements[0,0]
            initial_position = measurements[0,1:]
            initial_velocity = (measurements[1,1:] - measurements[0,1:]) / dt0

            # Run RIAE-AKF
            raie_akf = CoordsKF(dt0, initial_position, initial_velocity)
            t_list, mean_list, cov_list = raie_akf.riae_run(measurements)

            visualize_mean_cov.plot_points(mean_list,measurements[:,1:])
            visualize_mean_cov.plot_state_covariance_evolution_vs_time(t_list, cov_list)
        i += 1 