#!/usr/bin/env python

#########################
#Sample file to use robo_sim_listen.py
########################

import rospy
import sys

import roslib
roslib.load_manifest('project_simulation')
import time


from geometry_msgs.msg import *
from std_msgs.msg import *
from project_simulation.msg import *
from visualization_msgs.msg import *

import math
import copy
import tf

#publish Hz
PUB_RATE = 60


if __name__ == '__main__':
    rospy.init_node('test_robo')
    pub = rospy.Publisher('move_bin', project_simulation.msg.move_bin)
    
    temp_msg = project_simulation.msg.move_bin()
    time.sleep(1)
    #1
    temp_msg.bin_id = 13
    temp_msg.move_near_human = False
    pub.publish(temp_msg)

    time.sleep(8)

    #2
    temp_msg.bin_id = 3
    temp_msg.move_near_human = True
    pub.publish(temp_msg)

    time.sleep(8)

    #3
    temp_msg.bin_id = 11
    temp_msg.move_near_human = False
    pub.publish(temp_msg)