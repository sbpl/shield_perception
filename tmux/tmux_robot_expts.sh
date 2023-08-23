#!/bin/bash
tmux new-session -d -s 'robot'
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 0
tmux split-window -h
tmux select-pane -t 1
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v
tmux select-pane -t 4
tmux split-window -v
tmux select-pane -t 6
tmux split-window -v

tmux select-pane -t 0
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	send-keys 'roslaunch shield_perception zed_sync_nodelet.launch' \;

tmux select-pane -t 1
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
send-keys 'roslaunch shield_perception zed_filter.launch'

tmux select-pane -t 2
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
 	 send-keys 'rosrun shield_perception publish_tfs.py'

tmux select-pane -t 3
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	 send-keys 'rosrun shield_perception track_ball_zed.py'

tmux select-pane -t 4
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	 send-keys 'rosrun shield_planner shield_executive _real_robot:=true'

tmux select-pane -t 5
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	 send-keys 'roslaunch shield_planner preprocess_planner_abb.launch --screen'

tmux select-pane -t 6
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	 send-keys 'rviz -d ../rviz/zed2i.rviz'

tmux select-pane -t 7
# tmux send-keys 'export ROS_IP=10.68.0.102 ROS_MASTER_URI=http://10.68.0.1:11311' C-m
# tmux send-keys 'rosrun rqt_reconfigure rqt_reconfigure'
tmux send-keys 'source ../../../devel/setup.bash' Enter \; \
	 send-keys 'rostopic hz /zed2i/zed_nodelet/point_cloud/cloud_registered'

tmux -2 attach-session -d
