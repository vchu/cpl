#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2012, Georgia Institute of Technology
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

import roslib; roslib.load_manifest('visual_servo')
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist

from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose
import tf
import tf.transformations as tf_trans
import sys
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from visual_servo.srv import *
from array import *

def cleanup():
  rospy.loginfo('[IMPORTANT] Shutting Down. Set Twist to 0')
  pub = rospy.Publisher('l_cart/command', Twist)
  pub.publish(zero)
  rospy.sleep(0.5)
  pub.publish(zero)

def adjustVelocity(vel):
<<<<<<< HEAD
 clip_vel = 0.05
=======
 clip_vel = 0.12
>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
 ret = vel 
 if ret > clip_vel:
   ret = clip_vel
 elif -ret > clip_vel:
   ret = -clip_vel
 return ret 


 
def handle_move_request(req):
<<<<<<< HEAD
    pub = rospy.Publisher('l_cart/command', Twist)
=======
    pub = rospy.Publisher('r_cart/command', Twist)
>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
    t = req.twist # service call
    msg = Twist() # this is the message to the twist controller
    try:
      msg.linear.x = adjustVelocity(t.twist.linear.x)
      msg.linear.y = adjustVelocity(t.twist.linear.y)
      msg.linear.z = adjustVelocity(t.twist.linear.z)
      msg.angular.x = adjustVelocity(t.twist.angular.x)
      msg.angular.y = adjustVelocity(t.twist.angular.y)
      msg.angular.z = adjustVelocity(t.twist.angular.z)
      rospy.loginfo('vx:%+.5f vy:%+.5f vz:%+.5f wx:%+.5f wy:%+.5f wz:%+.5f', msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z)
      pub.publish(msg)
    except rospy.ServiceException, e:
      pub.publish(zero)
<<<<<<< HEAD

    return VisualServoTwistResponse()
         


=======

    return VisualServoTwistResponse()
         


>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
class VisualServoExecutionNode:
  def initial_arm(self, cs):
    rospy.loginfo('Setting to Initial msg...')
    cs.carefree_switch('l', '%s_arm', '$(find visual_servo)/params/rmc_joint_trajectory_params.yaml' )
    rospy.sleep(0.5)
    pub = rospy.Publisher('l_arm/command', JointTrajectory)
    msg = JointTrajectory()
    pts = JointTrajectoryPoint()
<<<<<<< HEAD
    pts.positions = array('d', [0.323, -0.6460, 0.00,
        -1.273, -10.512, 0.0570, 0.163])
 
    pts.time_from_start = rospy.Duration(3.0)
    msg.points = [pts]
   
    msg.joint_names = ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint','l_elbow_flex_joint', 'l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint']
    # msg.joint_names = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint','r_elbow_flex_joint', 'r_forearm_roll_joint','r_wrist_flex_joint','r_wrist_roll_joint']
=======
    pts.positions = array('d', [-1.32734204881265387, -0.64601608409943324, -1.4620635485239604,
                                      -1.2729772622637399, -10.5123303230158518, 0.0570651396529178, 0.163787989862169])
    pts.time_from_start = rospy.Duration(3.0)
    msg.points = [pts]
    msg.joint_names = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint',
                       'r_elbow_flex_joint', 'r_forearm_roll_joint','r_wrist_flex_joint','r_wrist_roll_joint']
>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
    # pub.publish(msg)
    rospy.sleep(0.5)
    # pub.publish(msg)
    rospy.sleep(3)

<<<<<<< HEAD
    pts.positions = array('d', [0.35, 0.167, 1.604, -1.89,
      -15.25, -1.282, 1.72])
 
=======
    pts.positions = array('d', [-0.750, +0.237, -1.60411927, 
                                -2.25, -15.85, -1.282, -1.72])
>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
    msg.points = [pts]
    pub.publish(msg)
    rospy.sleep(0.5)
    pub.publish(msg)
    
    rospy.sleep(5) 
    cs.carefree_switch('l', '%s_cart', '$(find visual_servo)/params/rmc_cartTwist_params.yaml')
    rospy.sleep(0.5)
    pub = rospy.Publisher('l_cart/command', Twist)
    pub.publish(zero)
    rospy.sleep(0.5)
    pub.publish(zero)

if __name__ == '__main__':
  try:
    node = VisualServoExecutionNode()
    rospy.init_node('vs_execute_node', log_level=rospy.DEBUG)
    global zero
    global service
    zero = Twist()
    zero.linear.x = 0.0
    zero.linear.y = 0.0
    zero.linear.z = 0.0
    zero.angular.x = 0.0
    zero.angular.y = 0.0
    zero.angular.z = 0.0
    rospy.on_shutdown(cleanup)
    
    cs = ControllerSwitcher()
    node.initial_arm(cs)
#    service = rospy.ServiceProxy('visual_servo_twist', VisualServoTwist)
    
    rospy.loginfo('Done initializing... Now advertise the Service')
    s = rospy.Service('movearm', VisualServoTwist , handle_move_request)
<<<<<<< HEAD
    rospy.loginfo('Ready to move arm')
=======
>>>>>>> 04206a0d0c433a3c0ca22cced7672835d0b6a8a5
    rospy.spin()
  
  except rospy.ROSInterruptException: pass


