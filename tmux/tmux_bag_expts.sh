#!/bin/zsh
tmux new-session -d -s 'bag'
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
tmux send-keys 'export ROS_IP=127.0.0.1 ROS_MASTER_URI=http://127.0.0.1:11311' C-m  
tmux send-keys 'roscore' C-m

tmux select-pane -t 1
tmux send-keys 'sleep 5; rosparam set use_sim_time true' C-m
tmux send-keys 'rosbag play --clock '

tmux select-pane -t 2
tmux send-keys 'rosrun shield_perception publish_tfs.py'

tmux select-pane -t 3
tmux send-keys 'rosrun shield_perception compute_projectile.py'

tmux select-pane -t 4
tmux send-keys 'rosrun object_detection subImg'

# tmux select-pane -t 5
# tmux send-keys 'roslaunch shield_planner preprocess_planner.launch --screen'

tmux select-pane -t 6
tmux send-keys 'rviz'

# tmux select-pane -t 7
# tmux send-keys 'rosrun rqt_reconfigure rqt_reconfigure'

tmux -2 attach-session -d