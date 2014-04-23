#!/usr/bin/python
#
#
# Copyright (c) 2013, Georgia Tech Research Corporation
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# \authors: Marc Killpack (Healthcare Robotics Lab, Georgia Tech.)
# \adviser: Charles Kemp (Healthcare Robotics Lab, Georgia Tech.)
#
# Edited by Vivian Chu (Socially Intelligent Machines Lab, Georgia Tech.)
#           4/15/14 - small things to work with Curi instead of darci
#           4/16/14 - major changes to allow for multiple body part control


import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
import sys, time, os
import math, numpy as np
import time
from geometry_msgs.msg import Wrench
from geometry_msgs.msg import Twist
from m3ctrl_msgs.msg import M3JointCmd
from sensor_msgs.msg import JointState
import threading
import copy
from hrl_lib import transforms as tr
from collections import *
import curi_arm_kinematics as dak

# For move_arm
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface
import moveit_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

##
#Class DarciClient()
#gives interface in python for controller to be similar to other MPC stuff we've done on Cody,
#also exposes kinematics, IK, joint limits and the ability to recorde robot data.
#

class DarciClient():
   
    def __init__(self, arm = 'l', record_data = False):
        self.arm = arm

        # These are chain values set by MEKA
        (self.RIGHT_ARM, self.LEFT_ARM, self.TORSO,
         self.HEAD, self.RIGHT_HAND, self.LEFT_HAND) = range(6)
        self.ZLIFT = 6
        
        # Joint modes from Meka
        self.JOINT_MODE_ROS_THETA_GC = 2
        self.SMOOTHING_MODE_SLEW = 1
        self.stiffness_percent = 0.75 

        # All body parts we want to control in array 
        self.body_parts = [self.RIGHT_ARM, self.LEFT_ARM, 
                           self.HEAD, self.ZLIFT]

        # Setup a dictionary to store all body parts
        self.states = defaultdict(dict)

        # Initialize all body parts
        for part in self.body_parts:
            self.states[part]['joint_angles'] = None 
            self.states[part]['joint_velocities'] = None 
            self.states[part]['torque'] = None 
            self.states[part]['J_h'] = None 
            self.states[part]['time'] = None 
            self.states[part]['desired_joint_angles'] = None 
            self.states[part]['stiffness_percent'] = 0.75 
            self.states[part]['ee_force'] = None 
            self.states[part]['ee_torque'] = None
            self.states[part]['Jc_l'] = []
            self.states[part]['n_l'] = []
            self.states[part]['values_l'] = []

        # Setup both arms
        self.states[self.LEFT_ARM]['kinematics'] = dak.DarciArmKinematics('l')
        self.states[self.RIGHT_ARM]['kinematics'] = dak.DarciArmKinematics('r')

        # Only do the arm that is selected
        self.humanoid_pub = rospy.Publisher('/humanoid_command', M3JointCmd)
        self.lock = threading.RLock()

        # Values for the zlift
        self.zlift_pub = rospy.Publisher('/zlift_command', M3JointCmd)
        self.zlift_pos = None #initial setup

	self.test_move_arm_pub = rospy.Publisher('/r_arm_controller/command', JointTrajectory)

        '''
        self.joint_angles = None
        self.joint_velocities = None
        self.torque = None
        self.J_h = None
        self.time = None
        self.desired_joint_angles = None
        self.stiffness_percent = 0.75
        self.ee_force = None
        self.ee_torque = None

        self.skins = None #will need code for all of these skin interfaces
        self.Jc_l = []
        self.n_l = []
        self.values_l = []
        '''
        #values from m3gtt repository in robot_config folder
        # These could be read in from a yaml file like this (import yaml; stream = open("FILE_NAME_HERE", 'r'); data = yaml.load(stream))
        # However, not clear if absolute path to yaml file (possibly on another computer) is better then just defining it here
        # The downside is obviously if someone tunes gains differently on the robot.
        self.joint_stiffness = (np.array([1, 1, 1, 1, 0.06, 0.08, 0.08])*180/math.pi*self.stiffness_percent).tolist()
        self.joint_damping = (np.array([0.06, 0.1, 0.015, 0.015, 0.0015, 0.002, 0.002])*180/math.pi*self.stiffness_percent).tolist()
       
        '''
        self.record_data = record_data

        if record_data == True:
            self.q_record = deque()
            self.qd_record = deque()
            self.torque_record = deque()
            self.times = deque()
        '''

        rospy.sleep(1.0)

        # Setup subscribers after a brief pause to setup joints
        self.state_sub = rospy.Subscriber('/humanoid_state', JointState, self.robotStateCallback)
        self.zlift_sub = rospy.Subscriber('/zlift_state', JointState, self.zliftStateCallback)

        # Wait for joint angle subscriber to fill joint_angles
        while self.states[self.HEAD]['joint_angles'] == None:
            rospy.sleep(0.01)

        for part in self.body_parts:
            self.states[part]['desired_joint_angles'] = copy.copy(self.states[part]['joint_angles'])

        # Setup the joint sending command for arm
        # Assumes some defaults (Joint modes, stiffness, etc.)
        joint_cmd = self.init_M3_Cmd(7, self.LEFT_ARM)
        joint_cmd.header.frame_id = 'humanoid_cmd_left'
        joint_cmd.chain = [self.LEFT_ARM]*7
        self.states[self.LEFT_ARM]['joint_cmd'] = joint_cmd
        
        joint_cmd = self.init_M3_Cmd(7, self.RIGHT_ARM)
        joint_cmd.header.frame_id = 'humanoid_cmd_right'
        joint_cmd.chain = [self.RIGHT_ARM]*7
        self.states[self.RIGHT_ARM]['joint_cmd'] = joint_cmd

        # Setup Zlift
        self.states[self.ZLIFT]['joint_cmd'] = self.init_M3_Cmd(1, self.ZLIFT)

        # Setup for head later if we want to do pointing and lookat
        # TODO: Fill no

        # Setup F/T subscribers for both hands
        self.ft_sub = rospy.Subscriber('/loadx6_left_state', Wrench, self.ftStateCallback)
        self.ft_sub = rospy.Subscriber('/loadx6_right_state', Wrench, self.ftStateCallback)

        # Setup for visualizing path
        self.display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)

        rospy.loginfo("Finished init")
       
    def init_M3_Cmd(self, size, part):
        m3_cmd = M3JointCmd()
        m3_cmd.header.stamp = rospy.Time.now()
        m3_cmd.chain_idx = np.arange(size,dtype=np.int16)
        m3_cmd.control_mode = [self.JOINT_MODE_ROS_THETA_GC]*size
        m3_cmd.stiffness = np.array([self.states[part]['stiffness_percent']]*size,dtype=np.float32) 
        m3_cmd.velocity = np.array([1.0]*size,dtype=np.float32) 
        m3_cmd.position = np.array(self.states[part]['joint_angles'],dtype=np.float32)
        m3_cmd.smoothing_mode = [0]*size

        return m3_cmd

    def zliftStateCallback(self,msg):
        self.lock.acquire()
        part = self.ZLIFT
        self.states[part]['joint_angles'] = copy.copy(msg.position[0])
        self.states[part]['joint_velocities'] = copy.copy(msg.velocity[0])
        self.states[part]['torque'] = copy.copy(msg.effort[0])
        self.states[part]['time'] = msg.header.stamp.secs + msg.header.stamp.nsecs*(1e-9)
        self.lock.release()

    def ftStateCallback(self,msg):
        self.lock.acquire()
        pos, rot = self.kinematics.FK(self.joint_angles)
        self.ee_force = (rot*np.matrix([msg.force.x, msg.force.y, msg.force.z]).reshape(3,1)).A1.tolist()
        self.ee_torque = (rot*np.matrix([msg.torque.x, msg.torque.y, msg.torque.z]).reshape(3,1)).A1.tolist()
        self.lock.release()

    def robotStateCallback(self, msg):
        self.lock.acquire()

        # Fill in the state for each body part
        # Kind of ugly... but for now use hard coded locations in msg
        for part in self.body_parts:
            if part == self.RIGHT_ARM:
                self.states[part]['joint_angles'] = copy.copy(msg.position[0:7])
                self.states[part]['joint_velocities'] = copy.copy(msg.velocity[0:7])
                self.states[part]['torque'] = copy.copy(msg.effort[0:7])
            if part == self.LEFT_ARM:
                self.states[part]['joint_angles'] = copy.copy(msg.position[7:14])
                self.states[part]['joint_velocities'] = copy.copy(msg.velocity[7:14])
                self.states[part]['torque'] = copy.copy(msg.effort[7:14])
            if part == self.HEAD:
                self.states[part]['joint_angles'] = copy.copy(msg.position[14:25])
                self.states[part]['joint_velocities'] = copy.copy(msg.velocity[14:25])
                self.states[part]['torque'] = copy.copy(msg.effort[14:25])
            self.states[part]['time'] = msg.header.stamp.secs + msg.header.stamp.nsecs*(1e-9)

        self.lock.release()

    def updateHapticState(self, body_part):
	    
        pos, rot = self.states[body_part]['kinematics'].FK(self.states[body_part]['joint_angles'])
        self.states[body_part]['end_effector_position'] = pos
        self.states[body_part]['end_effector_orient_quat'] = tr.matrix_to_quaternion(rot)
        self.states[body_part]['J_h'] = self.states[body_part]['kinematics'].Jacobian(self.states[body_part]['joint_angles'])

    def computeIK(self, poseMsg, body_part):
    
        # Currently doesn't work because q_init is not computed correctly
        # Convert the pose to what IK expects
        # TODO: Figure out transform frame
        pose = poseMsg.pose
        pos = np.ones((3,1)) # Expects 3x1 shape
        pos[0] = pose.position.x
        pos[1] = pose.position.y
        pos[2] = pose.position.z
        import pdb; pdb.set_trace()
        rot = tr.quaternion_to_matrix(np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z,pose.orientation.w]))

        self.states[body_part]['kinematics'].IK_vanilla(pos, rot, self.states[body_part]['joint_angles']) 

        return self.states[body_part]['kinematics'].IK(pos, rot) 

    def computeIK_moveit(self, poseMsg, body_part):
        
        # Uses move arm to do the planning
        scene = PlanningSceneInterface()
        robot = RobotCommander()
        rospy.sleep(1)
        if body_part == self.RIGHT_ARM:
            group = MoveGroupCommander("right_arm")
        elif body_part == self.LEFT_ARM:
            group = MoveGroupCommander("left_arm")
        else:
            rospy.loginfo("Given an invalid body part to plan: %d" % body_part)
            return None

        # Set just the position of the target currently
        # TODO: Fix the orientation
        group.set_position_target([poseMsg.pose.position.x,poseMsg.pose.position.y,poseMsg.pose.position.z])

        # Remove collision object?
        # collision_object = moveit_msgs.msg.
    
        # Init vars
        plan = RobotTrajectory()
        count = 1
        # Plan until we get a result - only 25 iterations currently
        while plan.joint_trajectory.points == [] and count < 20:
            print "Planning"
            plan = group.plan()

            # Actually excecute action
            group.execute(plan)

            # Keep track of attempts
            count = count + 1
            rospy.loginfo("Plan attempt: %d" % count)
        
        # return the plan
        return plan

    def computeIK_moveit_cart(self, poseMsg, body_part):

        # Uses move arm to do the planning
        scene = PlanningSceneInterface()
        robot = RobotCommander()
        rospy.sleep(1)
        if body_part == self.RIGHT_ARM:
            group = MoveGroupCommander("right_arm")
        elif body_part == self.LEFT_ARM:
            group = MoveGroupCommander("left_arm")
        else:
            rospy.loginfo("Given an invalid body part to plan: %d" % body_part)
            return None

        # Using cartesian paths.  Currently do not need for
        # just one point.  Maybe later for actual pushing vector?
        waypoints = [poseMsg.pose]

        eef_step = .01
        jump_thresh = 0
        avoid_collisions = True
        count = 0 
        best_fraction = 0
        fraction = 0
        best_plan = None
        while fraction < 0.9 and count < 30:
            (plan, fraction) = group.compute_cartesian_path(
                                     waypoints,   # waypoints to follow
                                     eef_step,        # eef_step
                                     jump_thresh,   # jump_threshold
                                     avoid_collisions)         # collisions
	    if fraction > best_fraction:
	        best_plan = plan
                best_fraction = fraction

	    count = count + 1

        # If we want to use the default controller rather than the move-it
        # controller to perform the action
        jtm = JointTrajectory()
        jtm.joint_names = plan.joint_trajectory.joint_names
        jtm.points = []
        for i in range(len(plan.joint_trajectory.points)):
            jtp = JointTrajectoryPoint()
            jtp.positions = plan.joint_trajectory.points[i].positions
            jtp.time_from_start = rospy.Duration(1.0)
            jtm.points.append(jtp)

        return jtm

    def updateSendCmd(self, body_part):

        bp_dict = self.states[body_part]
        cmd = bp_dict['joint_cmd']
        cmd.position = np.array(bp_dict['desired_joint_angles'], dtype=np.float32)

        if (body_part == self.ZLIFT):
            self.zlift_pub.publish(cmd)
        else:
            # Warn - this will send the last joint command again if never set by setDesiredPrior
            self.humanoid_pub.publish(cmd)

    def addDeltaToDesiredJointAngles(self, delta_angles):
        self.desired_joint_angles = (np.array(self.desired_joint_angles) + np.array(delta_angles)).tolist()

    def setDesiredJointAngles(self, angles, body_part):
        self.states[body_part]['desired_joint_angles'] = angles

    def getDesiredJointAngles(self, body_part):
        return self.states[body_part]['desired_joint_angles']

    def getJointAngles(self, body_part):
        return self.states[body_part]['joint_angles']

    def get_joint_angles(self, body_part):
        return self.getJointAngles(body_part)

    def getJointVelocities(self, body_part):
        return self.states[body_part]['joint_velocities']

    def recordCurData(self):
        if self.record_data == True:
            self.q_record.append(copy.copy(self.joint_angles))
            self.qd_record.append(copy.copy(self.joint_velocities))
            self.torque_record.append(copy.copy(self.torque))
            self.times.append(copy.copy(self.time))
        else:
            print "you didn't pass in the right flags to record robot data ... exiting"
            assert(False)

    def getRecordedData(self):
        return self.q_record, self.qd_record, self.torque_record, self.times

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.updateHapticState()
            self.updateSendCmd()

            # This will break currently
            if self.record_data == True:
                self.q_record.append(copy.copy(self.joint_angles))
                self.qd_record.append(copy.copy(self.joint_velocities))
                self.torque_record.append(copy.copy(self.torque))
                self.times.append(copy.copy(self.time))
            rate.sleep()

        # Also will break
        if self.record_data == True:
            import scipy.io as io
            data = {'q':self.q_record, 
                    'qd':self.qd_record,
                    'torque':self.torque_record,
                    'times':self.times}
            io.savemat('./darci_dynamics_data.mat', data)



if __name__ == '__main__':

    # this was for testing the torque values from the joints (it does not incorporate torque due to gravity)
    # however, it gives a good example of the different data and functions available to command the arm.
    # This does not assume new API using body_part, otherwise good example still

    rospy.init_node( 'move_arm_node', anonymous = True )
    darci = DarciClient(arm='l')
    rospy.sleep(5)
    inp = None
    while inp != 'q':
        #sending simple command to joints
        angles = [0.0]*7
        #angles[3] = math.pi/2

        # uploading command to arms through ROS
        darci.setDesiredJointAngles(angles)
        darci.updateSendCmd()

        # updating end effector position and orientation
        darci.updateHapticState()

        #getting joint, torque and ft data before test
        joint_angles = darci.joint_angles
        torque = darci.torque
        ee_torque = darci.ee_torque
        ee_force = darci.ee_force

        inp = raw_input('q for quit, otherwise will check diff in last torque value: \n')

        #getting joint, torque, end effector and ft data after test
        darci.updateHapticState()
        joint_angles_new = darci.joint_angles
        torque_new = darci.torque
        ee_torque_new = darci.ee_torque
        ee_force_new = darci.ee_force

        #comparing predicted torque change using stiffness (quasi-static, slow motions)
        predicted_change = np.matrix(np.diag(darci.joint_stiffness))*(np.matrix(joint_angles) - np.matrix(joint_angles_new)).reshape(7,1)
        actual_change = (np.matrix(torque_new) - np.matrix(torque)).reshape(7,1)
        print "predicted change in torque :\n", predicted_change
        print "actual change in torque :\n", actual_change

        #testing if transformation of force-torque data using FK is working
        diff_real_torque = np.array(ee_torque_new) - np.array(ee_torque)
        diff_real_force = np.array(ee_force_new) - np.array(ee_force)
        print "diff real torque is :\n", diff_real_torque/1000.
        print "diff real force is :\n", diff_real_force/1000.

        # checking what joint torques would be with no gravity given force at end effector
        # we can put the arm in configurations where certain joints are not affected by gravity
        # to use this.
        J_h_T = np.matrix(darci.J_h).T
        tau = J_h_T*np.matrix(np.hstack((diff_real_force, diff_real_torque))).reshape(6,1)/1000.
        print "torque at joints due to end effector force :\n", tau
        
    # going back to home position before quitting
    darci.setDesiredJointAngles([0]*7)
    darci.updateSendCmd()
