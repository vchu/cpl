#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <openni_tracker_msgs/jointData.h>
#include <openni_tracker_msgs/skeletonData.h>

using namespace std;
ros::Time last_timestamp;

void publishTransform(/* Argument for choosing which joint to send */)
{
	// Publish the fusioned data to TF

	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0)); // Add joint info
	transform.setRotation(tf::Quaternion(0.0, 0.0, 0.0)); // Add joint info

	// This is for rotation the data
	tf::Transform change_frame;
	change_frame.setOrigin(tf::Vector3(0, 0, 0)); // If we want to change the offset. 
	tf::Quaternion frame_rotation;
	frame_rotation.setEulerZYX(1.5708, 0, 1.5708); // The rotation This is 90 degrees on Z and X.
    change_frame.setRotation(frame_rotation);

    transform = change_frame * transform;
	

	br.sendTransform(tf::StampedTransform(transform, ros::Time::now()/* Not sure if time::now() is the best option*/
										 , "parent_name", "child_name"));
}

void msgCallback(const openni_tracker_msgs::skeletonData::ConstPtr &msg)
{
	// Retrieve the data from the two nodes and either save or fusion them.
	
	//cout << "The timestamp is: " << /*msg->header.stamp.toNSec()*/ ros::Time::now().toSec() << endl;
  ros::Time t=msg->header.stamp;
  ROS_INFO("time=%8f, kinectID: %d",t.toSec(),msg->kinectID);
  
/*	
	if(msg->kinect == 0)// Handle data from kinect 0
	{
		
		for(int k=0;k<=24;k++)// Looping through all the joints
	    {
	    	// We could just send each joint to the tf publisher from here
	    	openni_tracker_msgs::jointData joint = msg->joints[k];
	    
	    	joint.pose.position.x;
	    	joint.pose.position.y;
	    	joint.pose.position.z;
	    
	    	joint.pose.orientation.x;
	    	joint.pose.orientation.y;
	    	joint.pose.orientation.z;
	    	joint.pose.orientation.w;
	    	
		}
		
	}
	else if(msg->kinect == 1}// Handle data from kinect 1
	{
		
	}
*/

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "skeleton_fusion");
    ROS_INFO("Skeleton fusion initialized");
	ros::NodeHandle node;

    ros::Subscriber sub = node.subscribe("skeleton_data", 100, &msgCallback);


 	ros::spin();

	return 0;
}

