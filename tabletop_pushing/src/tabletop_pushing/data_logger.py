#!/usr/bin/env python
import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
import rosbag
import time
import os
import threading
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveGroupActionGoal
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Bool, Int8, String
from tabletop_pushing.msg import PushVector

class PushDataLogger(object):

    def __init__(self):
        
        # Internal cleanup and inits
        self.logger_flag = False
        rospy.on_shutdown(self.cleanup)
        self.lock = threading.Lock()
        self.lockBuffer = threading.Lock() # For message buffer

        # Setup a message buffer
        self.message_buffer = []
        
        # Subscribers
        rospy.loginfo("Setting up subscribers")
        rospy.Subscriber("push_data_logger_flag", Bool, self.callback)

        # Actual data we want
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_state_callback, queue_size = 100)
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback, queue_size = 100)
        rospy.Subscriber("/move_group/goal", MoveGroupActionGoal, self.move_group_cb, queue_size = 10)
        rospy.Subscriber("/push_vector", PushVector, self.push_vector_cb, queue_size = 10)
        rospy.Subscriber("/object_information", Pose2D, self.object_info_cb, queue_size = 10)
        rospy.Subscriber("/run_number", Int8, self.run_num_cb, queue_size = 10)
        rospy.Subscriber("/object_id", String, self.obj_num_cb, queue_size = 10)

    def callback(self, msg):
        # Setup flag from message
        rospy.loginfo(rospy.get_name() + ": I heard %s" % msg.data)
        self.logger_flag = msg.data

        # Create or close bag file
        if self.logger_flag is True:
            rospy.loginfo("Setting up bag file to write to")
            filename = "push_data_" + time.strftime("%Y-%m-%dT%H%M%S") + ".bag"
            rospy.loginfo("File name is: %s" % filename)
            self.bag = rosbag.Bag(os.path.join(os.path.expanduser("~"),'data','tabletop_pushing',filename),'w')


            # Created a new bag file - so write buffer to bag file
            # and reinit the buffer for next time
            # Lock this section to prevent writing to bag file before too early
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - Buffer")
            else:
                try:
                    for messages in self.message_buffer:
                        msg_type = messages[0]
                        #print msg_type
                        msg_obj = messages[1]
                        self.bag.write(msg_type, msg_obj)
                    self.message_buffer = []
                    
                    # write flag to file
                    self.bag.write('push_data_logger_flag',msg)
                
                finally:
                    self.lock.release()
            
        # Or else we're closing the bag file
        else:
            if hasattr(self,'bag'):
            	if not self.lock.acquire(): 
                   rospy.loginfo("Failed to acquire lock - Closing")
            	else:
                    try:
                        # write flag to file
                        self.bag.write('push_data_logger_flag',msg)

                        self.bag.close()
                	rospy.loginfo("Stopping recording")
                    finally:
                        self.lock.release()

    def gazebo_state_callback(self,msg):

        # Write audio stream to bag file
        if self.logger_flag is True:
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - gazebo/model_states")
            else:
                try:
                    self.bag.write('gazebo/model_states',msg)
                finally:
                    self.lock.release()

    def joint_state_callback(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - joint_states")
            else:
                try:
                    self.bag.write('joint_states',msg)
                finally:
                    self.lock.release() 
    
    def move_group_cb(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - move_group/goal")
            else:
                try:
                    self.bag.write('move_group/goal',msg)
                finally:
                    self.lock.release() 
            
    def push_vector_cb(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - push_vector")
            else:
                try:
                    self.bag.write('push_vector',msg)
                finally:
                    self.lock.release() 

    def object_info_cb(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - object_information")
            else:
                try:
                    self.bag.write('object_information',msg)
                finally:
                    self.lock.release() 

    def run_num_cb(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - run_number")
            else:
                try:
                    self.bag.write('run_number',msg)
                finally:
                    self.lock.release() 
                    
    def obj_num_cb(self,msg):
        
        # Write tf stream to bag file
        if self.logger_flag is True:
            # Currently blocking - relies on subscriber queue
            if not self.lock.acquire(): 
                rospy.loginfo("Failed to acquire lock - object_number")
            else:
                try:
                    self.bag.write('object_number',msg)
                finally:
                    self.lock.release() 

    def cleanup(self):
        # Close the file on exit in case early termination
        if hasattr(self,'bag'):
            self.bag.close()
            rospy.loginfo("Stopping recording")
        

    def store_message_buffer(self, msg, msg_type):
       
        # Check if we can write to buffer
        if not self.lockBuffer.acquire(): 
            rospy.loginfo("Failed to acquire lock - Buffer")
        else:
            try:
                # Check if the buffer is "full"
                if len(self.message_buffer) > 600: # This is 4 seconds worth of frames
                    self.message_buffer.pop(0) # get rid of the last message
                self.message_buffer.append((msg_type,msg))
            finally:
                self.lockBuffer.release()


def main():
    # Anonymous = true means that the node has a unique identifier
    rospy.init_node('push_data_logger', anonymous=True)
    rospy.loginfo("Logger Node Started")
    PushDataLogger()
    rospy.spin()

if __name__ == '__main__':
    main()
