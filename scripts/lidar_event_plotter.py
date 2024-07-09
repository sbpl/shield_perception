#! /usr/bin/env python

import rospy
import rosbag
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
import numpy as np
import matplotlib.pyplot as plt

bag = rosbag.Bag('/home/shield/scratch/ball_data/2023-08-15-20-18-33.bag')
event_thresh = 1.5

def LidarFilter(msg):
    # ceiling filter
    ranges = np.array(msg.ranges)
    filt_ranges = ranges[ranges < 2]

    scan_diff = np.linalg.norm(filt_ranges)
    return scan_diff

t0 = 0
inner_event = np.empty((0, 2))
outer_event = np.empty((0, 2))
first_iter = True
for topic, msg, t in bag.read_messages(topics=['/inner_scan', '/outer_scan']):
    if first_iter:
        t0 = t.to_sec()
        first_iter = False

    filt_res = LidarFilter(msg)
    if filt_res > event_thresh:
        if topic == '/inner_scan':
            inner_event = np.vstack((inner_event, np.array([t.to_sec()-t0, filt_res])))

        if topic == '/outer_scan':
            outer_event = np.vstack((outer_event, np.array([t.to_sec()-t0, filt_res])))
tF = inner_event[-1, 0]

print(np.shape(inner_event))
print(np.shape(outer_event))
plt.scatter(inner_event[:, 0], inner_event[:, 1])
plt.scatter(outer_event[:, 0], outer_event[:, 1])
plt.legend(['inner_scan', 'outer_scan'])
plt.xlabel('time (s)')
plt.ylabel('sum of ball distance (m)')
plt.grid()
# plt.grid(xdata=np.linspace(t0, tF, 30))
plt.show()

