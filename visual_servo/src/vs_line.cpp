/*********************************************************************
 *
 *  Copyright (c) 2012, Georgia Institute of Technology
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Georgia Institute of Technology nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/CvBridge.h>

// TF
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>

// Boost
#include <boost/shared_ptr.hpp>

// STL
#include <vector>
#include <deque>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <float.h>
#include <math.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND

// Others
#include <visual_servo/VisualServoTwist.h>
#include "visual_servo.cpp"

#define DEBUG_MODE 0

#define fmod(a,b) a - (float)((int)(a/b)*b)

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
/*
typedef struct {
  // pixels
  cv::Point image;
  // meters, note that xyz for this is different from the other one
  pcl::PointXYZ camera;
  // the 3d location in workspace frame
  pcl::PointXYZ workspace;
  pcl::PointXYZ workspace_angular;
} VSXYZ;
*/
using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PointStamped;
using visual_servo::VisualServoTwist;

/**
 * We are taping robot hand with three blue painter's tape 
 * and use those three features to do the image-based visual servoing
 * Uses Kinect as image sensor and provide a twist computation service
 */
class VisualServoNode
{
public:
  VisualServoNode(ros::NodeHandle &n) :
  n_(n), n_private_("~"),
  image_sub_(n, "color_image_topic", 1),
  depth_sub_(n, "depth_image_topic", 1),
  cloud_sub_(n, "point_cloud_topic", 1),
  sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
  it_(n), tf_(), have_depth_data_(false), camera_initialized_(false),
  desire_points_initialized_(false), below_threshold_count_(0)
  {
    vs_ = shared_ptr<VisualServo>(new VisualServo(JACOBIAN_TYPE_PSEUDO));
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    std::string default_optical_frame = "/openni_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
 
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);
    n_private_.param("num_downsamples", num_downsamples_, 2);
    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);
    
    n_private_.param("crop_min_x", crop_min_x_, 0);
    n_private_.param("crop_max_x", crop_max_x_, 640);
    n_private_.param("crop_min_y", crop_min_y_, 0);
    n_private_.param("crop_max_y", crop_max_y_, 480);
    n_private_.param("min_workspace_x", min_workspace_x_, -1.0);
    n_private_.param("min_workspace_y", min_workspace_y_, -1.2);
    n_private_.param("min_workspace_z", min_workspace_z_, -0.8);
    n_private_.param("max_workspace_x", max_workspace_x_, 1.75);
    n_private_.param("max_workspace_y", max_workspace_y_, 1.2);
    n_private_.param("max_workspace_z", max_workspace_z_, 0.6);
    
    // color segmentation parameters
    n_private_.param("tape_hue_value", tape_hue_value_, 137);
    n_private_.param("tape_hue_threshold", tape_hue_threshold_, 10);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 40);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 40);
    n_private_.param("default_val_value", default_val_value_, 200);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);
    
    // others
    n_private_.param("gain_vel", gain_vel_, 1.0);
    n_private_.param("gain_rot", gain_rot_, 1.0);
    n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    n_private_.param("term_threshold", term_threshold_, 0.0007);
    
    sync_.registerCallback(&VisualServoNode::sensorCallback, this);
  }

  /**
   * Called when Kinect information is avaiable. Refresh rate of about 100Hz 
   */
  void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg, 
                      const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    cv::Mat color_frame(bridge_.imgMsgToCv(img_msg));
    cv::Mat depth_frame(bridge_.imgMsgToCv(depth_msg));

    // Swap kinect color channel order
    cv::cvtColor(color_frame, color_frame, CV_RGB2BGR);

    XYZPointCloud cloud; 
    pcl::fromROSMsg(*cloud_msg, cloud);
    tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
        cloud.header.stamp, ros::Duration(0.5));
    pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
    prev_camera_header_ = cur_camera_header_;
    cur_camera_header_ = img_msg->header;

    cv::Mat workspace_mask(color_frame.rows, color_frame.cols, CV_8UC1,
                           cv::Scalar(255));

    // Black out pixels in color and depth images outside of workspace
    // As well as outside of the crop window
    for (int r = 0; r < color_frame.rows; ++r)
    {
      uchar* workspace_row = workspace_mask.ptr<uchar>(r);
      for (int c = 0; c < color_frame.cols; ++c)
      {
        // NOTE: Cloud is accessed by at(column, row)
        pcl::PointXYZ cur_pt = cloud.at(c, r);
        if (cur_pt.x < min_workspace_x_ || cur_pt.x > max_workspace_x_ ||
            cur_pt.y < min_workspace_y_ || cur_pt.y > max_workspace_y_ ||
            cur_pt.z < min_workspace_z_ || cur_pt.z > max_workspace_z_ ||
            r < crop_min_y_ || c < crop_min_x_ || r > crop_max_y_ ||
            c > crop_max_x_ )
        {
          workspace_row[c] = 0;
        }
      }
    }

    // focus only on the tabletop setting. do not care about anything far or too close
    color_frame.copyTo(cur_color_frame_, workspace_mask);
    cur_orig_color_frame_ = color_frame.clone();
    cur_depth_frame_ = depth_frame.clone();
    cur_point_cloud_ = cloud;

    // Store camera information only once
    if (!camera_initialized_)
    {
      cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(2.0));
      camera_initialized_ = true;
    }

    // initialize the desired points 
    if (!desire_points_initialized_)
    {
      orig_depth_ = depth_frame.clone(); 
      desired_locations_ = setDesiredPosition();
      if (desired_locations_.size() > 0)
      {
        prev_wp_ = desired_locations_.front();
        initializeService();
        desire_points_initialized_ = true;
        ROS_DEBUG("Desired Points Initialization Done...");
      }
    }
    else 
    {
      try
      {
        // compute the twist if everything is good to go
        visual_servo::VisualServoTwist srv = getTwist();

        // calling the service provider to move
        if (client_.call(srv))
        {
          // on success
        }
        else
        {
          // on failure
          ROS_WARN("Service FAILED...");
        }
      }
      catch(ros::Exception e)
      {
      }

    }
  }   
  
  float getError(std::vector<VSXYZ> a, std::vector<VSXYZ> b)
  {
    float e(0.0);
    unsigned int size = a.size() <= b.size() ? a.size() : b.size();
    if (size < 3)
      return 1000.0;
    for (unsigned int i = 0; i < size; i++)
    {
      pcl::PointXYZ a_c= a.at(i).camera;
      pcl::PointXYZ b_c= b.at(i).camera;
      e += pow(a_c.x - b_c.x ,2) + pow(a_c.y - b_c.y ,2);
    }
    return e;
  }
 
  void initializeService()
  {
    ROS_DEBUG("Hooking Up The Service");
    ros::NodeHandle n;
    client_ = n.serviceClient<visual_servo::VisualServoTwist>("movearm");
  }

  // Service method
  visual_servo::VisualServoTwist getTwist()
  {
    // initializing a zero service argument
    visual_servo::VisualServoTwist srv;
    srv.request.twist.twist.linear.x = 0; 
    srv.request.twist.twist.linear.y = 0; 
    srv.request.twist.twist.linear.z = 0;
    srv.request.twist.twist.angular.x = 0;
    srv.request.twist.twist.angular.y = 0; 
    srv.request.twist.twist.angular.z = 0;

    // if there are no more way points, terminate
    if (desired_locations_.size() <= 0)
    {
      ROS_INFO("Done with Servo");
      ros::shutdown();
      return srv;
    }
    // get all the blues 
    cv::Mat tape_mask = colorSegment(cur_color_frame_.clone(), 
        tape_hue_value_, tape_hue_threshold_);

    // find the three largest blues
    std::vector<cv::Moments> ms = findMoments(tape_mask, cur_color_frame_); 
    // order the blue tapes
    std::vector<cv::Point> pts = getMomentCoordinates(ms);
     
    // convert the features into proper form 
    std::vector<VSXYZ> features = PointToVSXYZ(cur_point_cloud_, cur_depth_frame_, pts);

    // Get the three points from a waypoint
    std::vector<VSXYZ> desired_vsxyz = getFeaturesFromXYZ(desired_locations_.front());

#define DISPLAY 0
#ifdef DISPLAY   
    // Draw the dots on image to be displayed
    cv::Point img_start_pt = prev_wp_.image;
    cv::Point img_end_pt = desired_locations_.front().image;
    cv::Point img_last_pt = desired_locations_.back().image;
    cv::line(cur_orig_color_frame_, img_start_pt, img_end_pt, cv::Scalar(72,255,0), 2);
    // cv::line(cur_orig_color_frame_, img_start_pt, img_last_pt, cv::Scalar(0,72,255), 2);

    cv::Point p = desired_locations_.front().image;
    for (unsigned int i = 1; i < desired_locations_.size(); i++)
    {
      cv::Point p = desired_locations_.at(i).image;
      cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(255, 158, 0), 1);
    }
    for (unsigned int i = 0; i < desired_vsxyz.size(); i++)
    {
      cv::Point p = desired_vsxyz.at(i).image;
      cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
      cv::circle(cur_orig_color_frame_, p, 1, cv::Scalar(0, 128, 255), 1);   
    }
    for (unsigned int i = 0; i < features.size(); i++)
    {
      cv::Point p = features.at(i).image;
      cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
    }
    
    cv::imshow("in", cur_orig_color_frame_); 
    cv::waitKey(display_wait_ms_);
    /* 
    ROS_DEBUG("------------ Hi -----------");
    for (unsigned int i = 0; i < desired_vsxyz.size(); i++)
    {
      printf("[D%d]\t", i); printVSXYZ(desired_vsxyz.at(i));
    }
    for (unsigned int i = 0; i < features.size(); i++)
    {
      printf("[F%d]\t", i); printVSXYZ(features.at(i));
    } 
    */
#endif

    // set the IM of desired locations for averaging
    if (JACOBIAN_TYPE_AVG == jacobian_type_)
      vs_->setDesiredInteractionMatrix(desired_vsxyz);

    srv = vs_->computeTwist(desired_vsxyz, features);

    // Error between desired pose and hand pose
    float e = getError(desired_vsxyz, features);
    srv.request.error = e;
    if(e < term_threshold_ && ++below_threshold_count_ >= 2)
    {
      ROS_DEBUG("Moving to New Waypoint");
      // poping the stack
      prev_wp_ = desired_locations_.front();
      desired_locations_.erase(desired_locations_.begin());
      below_threshold_count_ = 0;
    }

    return srv;
  }
 

  void printVSXYZ(VSXYZ i)
  {
    printf("Im: %+.3d %+.3d\tCam: %+.3f %+.3f %+.3f\twork: %+.3f %+.3f %+.3f\n",\
     i.image.x, i.image.y, i.camera.x, i.camera.y, i.camera.z, i.workspace.x, i.workspace.y, i.workspace.z);
  }


  
  cv::Mat pseudoInverse(cv::Mat im) 
  {
    return (im.t() * im).inv()*im.t();
  }
  
  float getZValue(cv::Mat depth_frame, int x, int y)
  {
    int window_size = 3;
    float value = 0;
    int size = 0; 
    for (int i = 0; i < window_size; i++) 
    {
      for (int j = 0; j < window_size; j++) 
      {
        // depth camera has x and y flipped. depth_frame.at(y,x)
        float temp = depth_frame.at<float>(y-(int)(window_size/2)+j, x-(int)(window_size/2)+i);
        // printf("[%d %d] %f\n", x-(int)(window_size/2)+i, y-(int)(window_size/2)+j, temp);
        if (!isnan(temp) && temp > 0 && temp < 2.0) 
        {
          size++;
          value += temp;
        }
      }
    }
    if (size == 0)
      return -1;
    return value/size;
  }
  
  
  /**
   * Still in construction:
   * to fix the orientation of the wrist, we now take a look at how the hand is positioned
   * and use it to compute the desired positions.
   */
  std::vector<VSXYZ> setDesiredPosition()
  {
    std::vector<pcl::PointXYZ> pts; pts.clear();
    pcl::PointXYZ start = cur_point_cloud_.at(cur_color_frame_.cols/2, cur_color_frame_.rows/2 + 90);
    pcl::PointXYZ end = cur_point_cloud_.at(cur_color_frame_.cols/2, cur_color_frame_.rows/2 - 50);
    
    // about 10 centimeters above the ground
    start.z += 0.35; 
    end.z += 0.25; 
    int steps = 8;

    float inc_x = (end.x - start.x) / steps;
    float inc_y = (end.y - start.y) / steps;
    float inc_z = (end.z - start.z) / steps;
    pts.push_back(start);
    for (int i = 0; i < steps; i++)
    {
      pts.push_back(pcl::PointXYZ(start.x + i*inc_x, start.y + i*inc_y, start.z + i*inc_z)); 
    }
    pts.push_back(end);
    std::vector<VSXYZ> v = Point3DToVSXYZ(pts);
    
    // test if desired points are all valid points
    if (JACOBIAN_TYPE_AVG == jacobian_type_)
    {
      for (unsigned int i = 0; i < v.size(); i++)
      {
        std::vector<VSXYZ> vsxyz = getFeaturesFromXYZ(v.at(i)); 
        if (!vs_->setDesiredInteractionMatrix(vsxyz))
        {
          v.clear();
          return v;
        }
      }
    }

    return v;
  }
  
  std::vector<VSXYZ> getFeaturesFromXYZ(VSXYZ origin_xyz)
  {
    pcl::PointXYZ origin = origin_xyz.workspace;
    std::vector<pcl::PointXYZ> pts; pts.clear();

    pcl::PointXYZ two = origin;
    pcl::PointXYZ three = origin;
    two.y -= 0.05; 
    three.x -= 0.05;
    
    pts.push_back(origin);
    pts.push_back(two);
    pts.push_back(three);
    return Point3DToVSXYZ(pts);
  }
  


  /************************************
   * PERCEPTION
   ************************************/
  /**
   * Take three biggest moments of specific color and returns 
   * the three biggest blobs or moments. This method assumes that
   * the features are in QR code like configuration
   * @param ms   All moments of color segmented
   * @return     returns vectors of cv::Point. Ordered in specific way 
   *             (1. top left, 2. top right, and 3. bottom left)
   **/
  std::vector<cv::Point> getMomentCoordinates(std::vector<cv::Moments> ms)
  {
    std::vector<cv::Point> ret;
    ret.clear();
    if (ms.size() == 3) { 
      double centroids[3][2];
      for (int i = 0; i < 3; i++) {
        cv::Moments m0 = ms.at(i);
        double x0, y0;
        x0 = m0.m10/m0.m00;
        y0 = m0.m01/m0.m00;
        centroids[i][0] = x0; 
        centroids[i][1] = y0; 
      }
      
      // find the top left corner using distance scheme
      cv::Mat vect = cv::Mat::zeros(3,2, CV_32F); 
      vect.at<float>(0,0) = centroids[0][0] - centroids[1][0];
      vect.at<float>(0,1) = centroids[0][1] - centroids[1][1];
      vect.at<float>(1,0) = centroids[0][0] - centroids[2][0];
      vect.at<float>(1,1) = centroids[0][1] - centroids[2][1];
      vect.at<float>(2,0) = centroids[1][0] - centroids[2][0];
      vect.at<float>(2,1) = centroids[1][1] - centroids[2][1];       
      
      double angle[3];
      angle[0] = abs(vect.row(0).dot(vect.row(1))); 
      angle[1] = abs(vect.row(0).dot(vect.row(2))); 
      angle[2] = abs(vect.row(1).dot(vect.row(2))); 
      
      // printMatrix(vect); 
      double min = angle[0]; 
      int one = 0;
      for (int i = 0; i < 3; i++)
      {
        // printf("[%d, %f]\n", i, angle[i]);
        if (angle[i] < min)
        {
          min = angle[i];
          one = i;
        }
      }
      
      // index of others depending on the index of the origin
      int a = one == 0 ? 1 : 0;
      int b = one == 2 ? 1 : 2; 
      // vectors of origin to a point
      double vX0, vY0, vX1, vY1, result;
      vX0 = centroids[a][0] - centroids[one][0];
      vY0 = centroids[a][1] - centroids[one][1];
      vX1 = centroids[b][0] - centroids[one][0];
      vY1 = centroids[b][1] - centroids[one][1];
      cv::Point pto(centroids[one][0], centroids[one][1]);
      cv::Point pta(centroids[a][0], centroids[a][1]);
      cv::Point ptb(centroids[b][0], centroids[b][1]);
      
      // cross-product: simplified assuming that z = 0 for both
      result = vX1*vY0 - vX0*vY1;
      ret.push_back(pto);
      if (result >= 0) {
        ret.push_back(ptb);
        ret.push_back(pta);
      }
      else {
        ret.push_back(pta);
        ret.push_back(ptb);
      }
    }
    return ret;
  } 
  
  
  std::vector<VSXYZ> Point3DToVSXYZ(std::vector<pcl::PointXYZ> in)  
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFrom3DPointToVSXYZ(in.at(i)));
    }
    return ret;
  }

  cv::Mat pointXYZToMat(pcl::PointXYZ in)
  {
    cv::Mat ret = cv::Mat::ones(4,1,CV_32F);
    ret.at<float>(0,0) = in.x;
    ret.at<float>(1,0) = in.y;
    ret.at<float>(2,0) = in.z;
    return ret;
  }
  
  pcl::PointXYZ matToPointXYZ(cv::Mat in)
  {
    return pcl::PointXYZ(in.at<float>(0,0), in.at<float>(1,0),
    in.at<float>(2,0));
  }

#ifdef SIMULATION
  // separate function for simulation since we can't utilize 
  // TF for rotation
  VSXYZ convertFrom3DPointToVSXYZ(pcl::PointXYZ in) 
  {
    // [X,Y,Z] -> [u,v] -> [x,y,z]
    VSXYZ ret;
    
    // 3D point in world frame
    pcl::PointXYZ in_c = in;

    // temporary to apply the transformation
    cv::Mat t = cv::Mat::ones(4,1, CV_32F);
    t.at<float>(0,0) = in_c.x;
    t.at<float>(1,0) = in_c.y;
    t.at<float>(2,0) = in_c.z;
    cv::Mat R = simulateGetRotationMatrix(sim_camera_x_, sim_camera_y_,
      sim_camera_z_, sim_camera_wx_, sim_camera_wy_, sim_camera_wz_);
    
    // apply the rotation
    t = R.inv() * t;
    in_c.x = t.at<float>(0,0);
    in_c.y = t.at<float>(1,0);
    in_c.z = t.at<float>(2,0);
    
    // really doesn't matter which frame they are in. ignored for now.
    cv::Point img = projectPointIntoImage(in_c, "/openni_rgb_optical_frame", "/openni_rgb_optical_frame");
    cv::Mat temp = projectImagePointToPoint(img);

    float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
    
    ret.image = img;
    // for simpler simulator, this will be the same
    ret.camera = _2d;
    ret.workspace= in;
    return ret;
  }

#else
  VSXYZ convertFrom3DPointToVSXYZ(pcl::PointXYZ in) 
  {
    // [X,Y,Z] -> [u,v] -> [x,y,z]
    VSXYZ ret;
    
    // 3D point in world frame
    pcl::PointXYZ in_c = in;
    cv::Point img = projectPointIntoImage(in_c, workspace_frame_, optical_frame_);
    cv::Mat temp = projectImagePointToPoint(img);

    float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
    
    ret.image = img;
    // for simpler simulator, this will be the same
    ret.camera = _2d;
    ret.workspace= in;
    return ret;
  }
#endif

  std::vector<VSXYZ> PointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, std::vector<cv::Point> in) 
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFromPointToVSXYZ(cloud, depth_frame, in.at(i)));
    }
    return ret;
  }
  
  VSXYZ convertFromPointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, cv::Point in) 
  {
    // [u,v] -> [x,y,z] (from camera intrinsics) & [X, Y, Z] (from PointCloud)
    VSXYZ ret;
    // pixel to meter value (using inverse of camera intrinsic) 
    cv::Mat temp = projectImagePointToPoint(in);

    // getZValue averages out z-value in a window to reduce noises
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), getZValue(depth_frame, in.x, in.y));
    pcl::PointXYZ _3d = cloud.at(in.x, in.y);
    
    ret.image = in;
    ret.camera = _2d;
    ret.workspace= _3d;
    return ret;
  } 
  
  
  /**
   * First, apply morphology to filter out noises and find contours around
   * possible features. Then, it returns the three largest moments
   * 
   * @param in  single channel image input
   * @param color_frame  need the original image for debugging and imshow
   * 
   * @return    returns ALL moment of specific color in the image
   **/
  std::vector<cv::Moments> findMoments(cv::Mat in, cv::Mat &color_frame) 
  {
    cv::Mat open, temp;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(in.clone(), open, cv::MORPH_OPEN, element);
    std::vector<std::vector<cv::Point> > contours; contours.clear();
    temp = open.clone();
    cv::findContours(temp, contours, cv::RETR_CCOMP,CV_CHAIN_APPROX_NONE);
    std::vector<cv::Moments> moments; moments.clear();
    
    for (unsigned int i = 0; i < contours.size(); i++) {
      cv::Moments m = cv::moments(contours[i]);
      if (m.m00 > min_contour_size_) {
        // first add the forth element
        moments.push_back(m);
        // find the smallest element of 4 and remove that
        if (moments.size() > 3) {
          double small(moments.at(0).m00);
          unsigned int smallInd(0);
          for (unsigned int j = 1; j < moments.size(); j++){
            if (moments.at(j).m00 < small) {
              small = moments.at(j).m00;
              smallInd = j;
            }
          }
          moments.erase(moments.begin() + smallInd);
        }
      }
    }
    return moments;
  }
  
  cv::Mat colorSegment(cv::Mat color_frame, int hue, int threshold)
  {
    /*
     * Often value = 0 or 255 are very useless. 
     * The distance at those end points get very close and it is not useful
     * Same with saturation 0. Low saturation makes everything more gray scaled
     * So the default setting are below 
     */
    return colorSegment(color_frame, hue - threshold, hue + threshold,  
        default_sat_bot_value_, default_sat_top_value_, 40, default_val_value_);
  }
  
  /** 
   * Very Basic Color Segmentation done in HSV space
   * Takes in Hue value and threshold as input to compute the distance in color space
   *
   * @param color_frame   color input from image
   *
   * @return  mask from the color segmentation 
   */
  cv::Mat colorSegment(cv::Mat color_frame, int _hue_n, int _hue_p, int _sat_n, int _sat_p, int _value_n,  int _value_p)
  {
    cv::Mat temp (color_frame.clone());
    cv::cvtColor(temp, temp, CV_BGR2HSV);
    std::vector<cv::Mat> hsv;
    cv::split(temp, hsv);
    
    // masking out values that do not fall between the condition 
    cv::Mat wm(color_frame.rows, color_frame.cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < temp.rows; r++)
    {
      uchar* workspace_row = wm.ptr<uchar>(r);
      for (int c = 0; c < temp.cols; c++)
      {
        // hue = 0 to 360 (degrees)
        // sat, value = 0 to 100 (percent)
        int hue   = 2*(int)hsv[0].at<uchar>(r, c);  
        float sat   = 0.392*(int)hsv[1].at<uchar>(r, c); // 0.392 = 100/255
        float value = 0.392*(int)hsv[2].at<uchar>(r, c);
        // printf("[%d, %d] hue = %d\n", r, c, hue);
        if (_hue_n < hue && hue < _hue_p)
          if (_sat_n < sat && sat < _sat_p)
            if (_value_n < value && value < _value_p)
              workspace_row[c] = 255;
      } 
    }
    
    // removing unwanted parts by applying mask to the original image
    return wm;
  }
 

  /**************************** 
   * Projection & Conversions
   ****************************/
  std::vector<VSXYZ> Point3DToVSXYZ(std::vector<pcl::PointXYZ> in)  
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFrom3DPointToVSXYZ(in.at(i)));
    }
    return ret;
  }

  VSXYZ convertFrom3DPointToVSXYZ(pcl::PointXYZ in) 
  {
    // [X,Y,Z] -> [u,v] -> [x,y,z]
    VSXYZ ret;
    
    // 3D point in world frame
    pcl::PointXYZ in_c = in;
    cv::Point img = projectPointIntoImage(in_c, workspace_frame_, optical_frame_);
    cv::Mat temp = projectImagePointToPoint(img);

    float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
    
    ret.image = img;
    // for simpler simulator, this will be the same
    ret.camera = _2d;
    ret.workspace= in;
    return ret;
  }

  std::vector<VSXYZ> PointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, std::vector<cv::Point> in) 
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFromPointToVSXYZ(cloud, depth_frame, in.at(i)));
    }
    return ret;
  }
  
  VSXYZ convertFromPointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, cv::Point in) 
  {
    // [u,v] -> [x,y,z] (from camera intrinsics) & [X, Y, Z] (from PointCloud)
    VSXYZ ret;
    // pixel to meter value (using inverse of camera intrinsic) 
    cv::Mat temp = projectImagePointToPoint(in);

    // getZValue averages out z-value in a window to reduce noises
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), getZValue(depth_frame, in.x, in.y));
    pcl::PointXYZ _3d = cloud.at(in.x, in.y);
    
    ret.image = in;
    ret.camera = _2d;
    ret.workspace= _3d;
    return ret;
  }

  
  /**
   * transforms a point in pixels to meter using the inverse of 
   * image intrinsic K
   * @param in a point to be transformed
   * @return returns meter value of the point in cv::Mat
   */ 
  cv::Mat projectImagePointToPoint(cv::Point in) 
  {
    if (K.rows == 0 || K.cols == 0) {
      // Camera intrinsic matrix
      K  = cv::Mat(cv::Size(3,3), CV_64F, &(cam_info_.K));
      K.convertTo(K, CV_32F);
    }
    
    cv::Mat k_inv = K.inv();
    
    cv::Mat mIn  = cv::Mat(3,1,CV_32F);
    mIn.at<float>(0,0) = in.x; 
    mIn.at<float>(1,0) = in.y; 
    mIn.at<float>(2,0) = 1; 
    return k_inv * mIn;
  }
  
  /**
   * transforms a cv::Mat in pixels to meter using the inverse 
   * Image Intrinsic K
   * @param in  cv::Mat input to be transformed
   * @return    returns meter in cv::Mat
   */ 
  cv::Mat projectImageMatToPoint(cv::Mat in)   
  { 
    cv::Point p(in.at<float>(0,0), in.at<float>(1,0));
    return projectImagePointToPoint(p);
  }
  
  /** 
   * Transforms a point in Point Cloud to Image Frame (pixels)
   * @param cur_point_pcl  The point to be transformed in pcl::PointXYZ
   * @param point_frame    The frame that the PointCloud is in
   * @param target_frame   Image frame
   * @return               returns the pixel value of the PointCloud in image frame
   */
  cv::Point projectPointIntoImage(pcl::PointXYZ cur_point_pcl,
                                  std::string point_frame, std::string target_frame)
  {
    PointStamped cur_point;
    cur_point.header.frame_id = point_frame;
    cur_point.point.x = cur_point_pcl.x;
    cur_point.point.y = cur_point_pcl.y;
    cur_point.point.z = cur_point_pcl.z;
    return projectPointIntoImage(cur_point, target_frame);
  }
  
  cv::Point projectPointIntoImage(PointStamped cur_point,
                                  std::string target_frame)
  {
    if (K.rows == 0 || K.cols == 0) {
      // Camera intrinsic matrix
      K  = cv::Mat(cv::Size(3,3), CV_64F, &(cam_info_.K));
      K.convertTo(K, CV_32F);
    }
    cv::Point img_loc;
    try
    {
      // Transform point into the camera frame
      PointStamped image_frame_loc_m;
      tf_->transformPoint(target_frame, cur_point, image_frame_loc_m);
      // Project point onto the image
      img_loc.x = static_cast<int>((K.at<float>(0,0)*image_frame_loc_m.point.x +
                                    K.at<float>(0,2)*image_frame_loc_m.point.z) /
                                   image_frame_loc_m.point.z);
      img_loc.y = static_cast<int>((K.at<float>(1,1)*image_frame_loc_m.point.y +
                                    K.at<float>(1,2)*image_frame_loc_m.point.z) /
                                   image_frame_loc_m.point.z);
    }
    catch (tf::TransformException e)
    {
      ROS_ERROR_STREAM(e.what());
    }
    return img_loc;
  }

  /**********************
   * HELPER METHODS
   **********************/
  void printMatrix(cv::Mat_<double> in)
  {
    for (int i = 0; i < in.rows; i++) {
      for (int j = 0; j < in.cols; j++) {
        printf("%+.5f\t", in(i,j)); 
      }
      printf("\n");
    }
  }
  
  float getZValue(cv::Mat depth_frame, int x, int y)
  {
    int window_size = 3;
    float value = 0;
    int size = 0; 
    for (int i = 0; i < window_size; i++) 
    {
      for (int j = 0; j < window_size; j++) 
      {
        // depth camera has x and y flipped. depth_frame.at(y,x)
        float temp = depth_frame.at<float>(y-(int)(window_size/2)+j, x-(int)(window_size/2)+i);
        // printf("[%d %d] %f\n", x-(int)(window_size/2)+i, y-(int)(window_size/2)+j, temp);
        if (!isnan(temp) && temp > 0 && temp < 2.0) 
        {
          size++;
          value += temp;
        }
      }
    }
    if (size == 0)
      return -1;
    return value/size;
  }
 
  /**
   * Executive control function for launching the node.
   */
  void spin()
  {
    while(n_.ok())
    {
      ros::spinOnce();
    }
  }
  
protected:
  ros::NodeHandle n_;
  ros::NodeHandle n_private_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  message_filters::Synchronizer<MySyncPolicy> sync_;
  image_transport::ImageTransport it_;
  sensor_msgs::CameraInfo cam_info_;
  sensor_msgs::CvBridge bridge_;
  shared_ptr<tf::TransformListener> tf_;
  cv::Mat cur_color_frame_;
  cv::Mat cur_orig_color_frame_;
  cv::Mat cur_depth_frame_;
  cv::Mat cur_workspace_mask_;
  
  cv::Mat orig_depth_;
  VSXYZ prev_wp_;

  std_msgs::Header cur_camera_header_;
  std_msgs::Header prev_camera_header_;
  XYZPointCloud cur_point_cloud_;

  bool have_depth_data_;
  int display_wait_ms_;
  int num_downsamples_;
  std::string workspace_frame_;
  std::string optical_frame_;
  bool camera_initialized_;
  bool desire_points_initialized_;
  std::string cam_info_topic_;
  int tracker_count_;

  // filtering 
  double min_workspace_x_;
  double max_workspace_x_;
  double min_workspace_y_;
  double max_workspace_y_;
  double min_workspace_z_;
  double max_workspace_z_;
  int crop_min_x_;
  int crop_max_x_;
  int crop_min_y_;
  int crop_max_y_;

  // segmenting
  int tape_hue_value_;
  int tape_hue_threshold_;
  int default_sat_bot_value_;
  int default_sat_top_value_;
  int default_val_value_;
  double min_contour_size_;

  // others    
  shared_ptr<VisualServo> vs_;
  int jacobian_type_;
  double gain_vel_;
  double gain_rot_;
  double term_threshold_;
  unsigned int below_threshold_count_;

  cv::Mat desired_jacobian_;
  std::vector<VSXYZ> desired_locations_;
  // std::vector<pcl::PointXYZ> desired_locations_;
  cv::Mat K;

  ros::ServiceServer twistServer;
  ros::ServiceClient client_;
};

int main(int argc, char ** argv)
{
  srand(time(NULL));
  ros::init(argc, argv, "visual_servo_line");
  
  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  
  ros::NodeHandle n;
  VisualServoNode vs_node(n);
/*
  cv::Mat m = cv::imread("/home/a/a.png", 1);
  if (!m.data)
  {
    ROS_ERROR("I GOT NOTHING!");
    return -1;
  }
  
  // segment the right color
  cv::Mat mask = vs_node.colorSegment(m.clone(), 120, 20);
  cv::Mat mm;
  m.copyTo(mm, mask);

 
  cv::imshow("in", m); 
  cv::imshow("masked", mm);
  cv::waitKey(0);
*/
  vs_node.spin();
  return 0;
}