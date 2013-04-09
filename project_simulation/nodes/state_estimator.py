#!/usr/bin/env python
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

'''
Collects following information:
- Probability distribution from the inference engine
- Bin - alvar_markers or bin-id & loc-names
- endfactor position, as well as if its carrying a bin(bin_id in that case)
'''

def collect_info():


def pub_as_one():


if __name__ == '__main__':

    rospy.init_node('state_estimator')

    #bin positions
    sub_bins = rospy.Subscriber('ar_pose_marker', 
                                project_simulation.msg.bins_loc, ar_listen)
    
    #end-factor position
    sub_endf = rospy.Subscriber('endfactor_pose', geometry_msgs.msg.PoseStamped,
                                endf_listen)
    
    #TODO: Inference probability distribution
    sub_inf_prob = rospy.Subscriber('inference_dist', proj_simul.msg.ProbDist, 
                                    inf_prob_listen)
    
    #publish as one
    pub_cur_state = rospy.Publisher('current_state')
    
    while not rospy.shutdown():
        collect_info()
        pub_as_one()
    
