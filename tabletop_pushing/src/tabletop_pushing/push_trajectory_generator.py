#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2013, Georgia Institute of Technology
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#  * Neither the name of the Georgia Institute of Technology nor the names of
#     its contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, Pose2D
from tabletop_pushing.srv import *
from tabletop_pushing.msg import *
from math import copysign, pi, ceil
import svmutil
import numpy as np

class PiecewiseLinearTrajectoryGenerator:
    '''
    Define a finer time scale trajectory through a list of fixed points
    '''
    def __init__(self, max_step_size, min_num_segment_steps=1):
        self.max_step_size = max_step_size
        self.min_num_segment_steps = min_num_segment_steps

    def generate_trajectory(self, start_pose, pose_list):
        '''
        start_pose - Pose2D() location of start pose for trajectory
        pose_list - list of Pose2D() locations for the rest of the trajectory
        '''
        start_loc = np.array([start_pose.x, start_pose.y])#, start_pose.theta])
        trajectory = [start_loc]
        for i in xrange(len(pose_list)):
            p_i = np.array([pose_list[i].x, pose_list[i].y])#, pose_list[i].theta])

            num_steps = int(ceil(np.linalg.norm(p_i[:2] - trajectory[-1][:2])/self.max_step_size))
            num_steps = max(num_steps, self.min_num_segment_steps)

            step = (p_i - trajectory[-1]) / num_steps

            # print 'step =',step
            for j in xrange(num_steps):
                next_loc = trajectory[-1] + step
                trajectory.append(next_loc)
        return trajectory

class StraightLineTrajectoryGenerator:
    def __init__(self, max_step_size, min_num_steps=1):
        self.pltg = PiecewiseLinearTrajectoryGenerator(max_step_size, min_num_steps)

    def generate_trajectory(self, start_pose, end_pose):
        '''
        start_pose - Pose2D() location of start pose for trajectory
        end_pose - Pose2D() location for the final point of the trajectory
        '''
        return self.pltg.generate_trajectory(start_pose, [end_pose])

class ViaPointTrajectoryGenerator:
    '''
    Generates a smooth (parabolic) transitioned trajectory of straight lines, following the method of Russ Taylor
    '''

    def generate_trajectory(self, H, start_pose, pose_list, tau = 0.25):
        '''
        H - Number of time steps to get from start_pose to the final pose in pose_list
        start_pose - Pose2D() location of start pose for trajectory
        pose_list - list of Pose2D() locations for the rest of the trajectory
        tau - time before via time at which the curve starts, defaults to 0.25
        '''
        # TODO: Add in max step size
        num_line_steps = H/len(pose_list)
        delta_t = 1.0/num_line_steps

        start_loc = np.array([start_pose.x, start_pose.y]) #, start_pose.theta])
        trajectory = [start_loc]

        # TODO: Add smooth rotational transitions

        for i in xrange(len(pose_list)):
            p0 = trajectory[-1]
            p1 = np.array([pose_list[i].x, pose_list[i].y])#, pose_list[i].theta])
            delta_p1 = p1 - p0
            if i < len(pose_list) - 1:
                p2 = np.array([pose_list[i+1].x, pose_list[i+1].y])#, pose_list[i+1].theta])
                delta_p2 = p2 - p1
            if i > 0:
                pn1 = np.array([pose_list[i-1].x, pose_list[i-1].y])#, pose_list[i-1].theta])
                delta_p0 = p0 - pn1

            for j in xrange(num_line_steps):
                t = delta_t*(j+1)
                if t < 1.0-tau:
                    # Straight line, always for first segment, after tau for subsequent segments
                    if i == 0 or t > tau:
                        p = p1 - (1.0-t)*delta_p1
                    else: # Smoothed
                        p = p0 - delta_p0/(4.0*tau)*(t-tau)**2 + delta_p1/(4.0*tau)*(t+tau)**2
                elif i < len(pose_list) - 1: # Smoothed end of line
                    p = p1 - delta_p1/(4.0*tau)*(t-1.0-tau)**2 + delta_p2/(4.0*tau)*(t-1.0+tau)**2
                else: # End of last line segment
                    p = p1 - (1.0-t)*delta_p1
                trajectory.append(p)
        return trajectory
