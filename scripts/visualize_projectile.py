#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from shield_planner_msgs.msg import Projectile


rospy.init_node('shield_perception_bound')
pub = rospy.Publisher("/projectile_animation_vis",
                     Marker,
                     queue_size=1)

def visualize_projectile_points( bpes, pmmsg ):
    # markers = MarkerArray()
    # for ind, es in enumerate(bpes):
    marker = Marker()
    marker.header.frame_id = "odom_combined"
    marker.type = marker.SPHERE
    # marker.id = ind
    marker.id = 0
    marker.action = marker.ADD
    marker.scale.x = 0.15
    marker.scale.y = 0.15
    marker.scale.z = 0.15
    marker.color.a = 0.8
    marker.color.r = 0.5
    marker.color.g = 1.0
    marker.color.b = 0.8
    marker.pose.orientation.w = 1.0
    marker.pose.position.x = bpes[0]
    marker.pose.position.y = bpes[1] 
    marker.pose.position.z = bpes[2]
        # markers.markers.append(marker)
    # pmmsg.publish(markers)
    pmmsg.publish(marker)

def projectile_callback( proj ):
    rospy.loginfo("In projectile vis callback")
    x_init = proj.position.x
    y_init = proj.position.y
    z_init = proj.position.z
    vx_init = proj.velocity.x
    vy_init = proj.velocity.y
    vz_init = proj.velocity.z
    t_init = proj.header.stamp
    x = x_init
    y = y_init
    z = z_init
    t_delt = 0
    r = rospy.Rate(1000)
    while z >= 0:
        t_cur = rospy.Time.now().to_sec()
        t_delt = t_cur - t_init.to_sec();
        t_squared = t_delt**2
        x = x_init + vx_init*t_delt
        y = y_init + vy_init*t_delt
        z = z_init + vz_init*t_delt - 0.5 * 9.81 * t_squared
        visualize_projectile_points([x,y,z], pub)
        r.sleep()
    rospy.logwarn("Projectile vis ends in {}".format(t_delt))


def visiualizeBall():
    rospy.Subscriber("projectile", Projectile,
                    projectile_callback,
                    queue_size=1)

    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    visiualizeBall()
