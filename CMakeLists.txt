cmake_minimum_required(VERSION 3.0.2)
project(shield_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  visualization_msgs
  rospy)

catkin_package(
  CATKIN_DEPENDS shield_planner_msgs)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

# add_executable(fake_perception_node src/fake_perception_node.cpp)
# target_link_libraries(fake_perception_node ${catkin_LIBRARIES})
