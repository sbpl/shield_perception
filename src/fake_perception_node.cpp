#include <iostream>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <shield_planner_msgs/LineSegmentId.h>
#include <shield_planner_msgs/Projectile.h>

int main(int argc, char **argv){

    ros::init(argc, argv, "fake_perception_node");

    ros::NodeHandle nh;

    ros::Publisher proj_path = nh.advertise<shield_planner_msgs::Projectile>("/projectile", 1);
    
    shield_planner_msgs::Projectile proj_msg;

    proj_msg.header.stamp = ros::Time::now();
    proj_msg.position.x =  9.31469501586;
    proj_msg.position.y =  -1.43852786901;
    proj_msg.position.z =  0.0;

    proj_msg.velocity.x = -7.69668769836;
    proj_msg.velocity.y = 1.59418177605;
    proj_msg.velocity.z = 5.63860082626;

    ros::Duration(0.5).sleep();
    // shield_planner_msgs::Projectile proj_msg2 = proj_msg;
    proj_msg2.velocity.z += 0.5;
    proj_msg2.velocity.x += 1.0;

    while (ros::ok()) {
        ROS_INFO("First projectile");
        proj_path.publish(proj_msg);
        ros::Duration(0.2).sleep();
        
        ROS_INFO("Second projectile");
        proj_path.publish(proj_msg2);
        ROS_INFO("First & Second projectile");
        getchar();
        
    }

    return 0;

}
