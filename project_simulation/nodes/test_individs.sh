#MPVL_LD_SPL_LD
rosparam set /prob_percept_screw 0.0
rosparam set /add_hand_off 0.0002
rosparam set /add_duration 0.0

seed=1

sleep 5

rosrun project_simulation fix_tf.py &
PID1=$!

rosrun project_simulation ar_markers_publish.py &
PID2=$!

rosrun project_simulation robo_sim_listen.py &
PID3=$!

echo y | rosrun project_simulation hands_sim linear_chain_1 n $seed &
PID4=$!

rosrun airplane_assembly_inference_0313 planning_from_matlab.py &
PID5=$!

rosrun airplane_assembly_inference_0313 inference_from_matlab.py

kill $PID1
kill $PID2
kill $PID3
kill $PID4
kill $PID5


echo END

sleep 5

