#!/usr/bin/env python

from KBHit import KBHit
import rospy
import ros_numpy as rnp
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# projectile imports 
from shield_planner_msgs.srv import TestbedProjectile
from shield_planner_msgs.msg import Projectile
from datetime import datetime
import math
import rospkg

sc_projectile = None
ki_projectile = None
received_ki_projectile = False
received_sc_projectile = False
transformation = None
file_counter = 0

rospack = rospkg.RosPack()

def sc_callback(msg):
    global sc_projectile
    global received_sc_projectile
    print("got sc msg")
    received_sc_projectile = True
    sc_projectile = msg

def ki_callback(msg):
    global ki_projectile
    global received_ki_projectile
    if (not np.isnan(msg.velocity.x)) and  \
        (not np.isnan(msg.velocity.y)) and \
        (not np.isnan(msg.velocity.z)):
        print("got ki msg")
        received_ki_projectile = True
        ki_projectile = msg

def logProjectileData():
    print("logProjectileData")

    print(sc_projectile.position.x)
    print(sc_projectile.position.y)
    print(sc_projectile.position.z)
    print(sc_projectile.velocity.x)
    print(sc_projectile.velocity.y)
    print(sc_projectile.velocity.z)

    projectile_array = np.array([[sc_projectile.position.x, 
                                    sc_projectile.position.y, 
                                    sc_projectile.position.z], 
                                   [sc_projectile.velocity.x, 
                                    sc_projectile.velocity.y, 
                                    sc_projectile.velocity.z], 
                                   [ki_projectile.position.x, 
                                    ki_projectile.position.y, 
                                    ki_projectile.position.z], 
                                   [ki_projectile.velocity.x, 
                                    ki_projectile.velocity.y, 
                                    ki_projectile.velocity.z]])

    print("projectile_array ", projectile_array)
    
    # logging_file = open(rospack.get_path('shield_perception')+"/logs/projectile_throw_"+datetime.strftime("%Y%m%d_%H%M%S")+".npy", "w+")
    logging_file = open(rospack.get_path('shield_perception')+"/logs/projectile_throw_"+str(file_counter)+".npy", "w+")
    
    np.save(logging_file, projectile_array)
    # np.savez(logging_file)


    # print("i'm here ")
    # import ipdb; ipdb.set_trace
    
def listener():

    global received_ki_projectile
    global received_sc_projectile
    global file_counter

    rospy.init_node('log_projectile_data', anonymous=True)
    rospy.Subscriber("projectile", Projectile, sc_callback)
    rospy.Subscriber("projectile_2", Projectile, ki_callback)

    kb = KBHit()
    print('Hit ESC to save')

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        print("file count0 ", file_counter)
        if kb.kbhit():
            print("@")
            c = kb.getch()
            print("c = ", c)
            print("ord c = ", ord(c))
            
            if ord(c) == 27: # ESC
                print("saving file now ")
                if (received_ki_projectile == True) and (received_sc_projectile == True):
                    global file_counter 
                    logProjectileData()
                    file_counter += 1
                    received_sc_projectile = False
                    received_ki_projectile = False

                # break
            # print(c)

        # rospy.spin()
        rate.sleep()
    
    kb.set_normal_term()


if __name__ == '__main__':
    listener()