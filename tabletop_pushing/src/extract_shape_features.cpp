#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/ros/conversions.h>
#include <pcl/common/pca.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <cpl_visual_features/helpers.h>
#include <cpl_visual_features/features/kernels.h>
#include <time.h> // for srand(time(NULL))

// libSVM
#include <libsvm/svm.h>

using namespace cpl_visual_features;
using namespace tabletop_pushing;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;

#define XY_RES 0.001
#define USE_HKS_DESCRIPTOR 1

ShapeLocations start_loc_history_;
double start_loc_arc_length_percent_;
int start_loc_push_sample_count_;
XYZPointCloud hull_cloud_;
PushTrackerState cur_state_;
double point_cloud_hist_res_ = 0.01;
inline int objLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

inline double sqrDistXY(pcl::PointXYZ a, pcl::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  return dx*dx+dy*dy;
}

pcl::PointXYZ worldPointInObjectFrame(pcl::PointXYZ world_pt, PushTrackerState& cur_state)
{
  // Center on object frame
  pcl::PointXYZ shifted_pt;
  shifted_pt.x = world_pt.x - cur_state.x.x;
  shifted_pt.y = world_pt.y - cur_state.x.y;
  shifted_pt.z = world_pt.z - cur_state.z;
  double ct = cos(cur_state.x.theta);
  double st = sin(cur_state.x.theta);
  // Rotate into correct frame
  pcl::PointXYZ obj_pt;
  obj_pt.x =  ct*shifted_pt.x + st*shifted_pt.y;
  obj_pt.y = -st*shifted_pt.x + ct*shifted_pt.y;
  obj_pt.z = shifted_pt.z; // NOTE: Currently assume 2D motion
  return obj_pt;
}

pcl::PointXYZ objectPointInWorldFrame(pcl::PointXYZ obj_pt, PushTrackerState& cur_state)
{
  // Rotate out of object frame
  pcl::PointXYZ rotated_pt;
  double ct = cos(cur_state.x.theta);
  double st = sin(cur_state.x.theta);
  rotated_pt.x = ct*obj_pt.x - st*obj_pt.y;
  rotated_pt.y = st*obj_pt.x + ct*obj_pt.y;
  rotated_pt.z = obj_pt.z;  // NOTE: Currently assume 2D motion
  // Shift to world frame
  pcl::PointXYZ world_pt;
  world_pt.x = rotated_pt.x + cur_state.x.x;
  world_pt.y = rotated_pt.y + cur_state.x.y;
  world_pt.z = rotated_pt.z + cur_state.z;
  return world_pt;
}

static inline double dist(pcl::PointXYZ a, pcl::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double sqrDist(pcl::PointXYZ a, pcl::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return dx*dx+dy*dy+dz*dz;
}

ShapeLocation chooseFixedGoalPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, bool new_object,
                                          int num_start_loc_pushes_per_sample, int num_start_loc_sample_locs)
{
  double hull_alpha = 0.01;
  XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);
  hull_cloud_ = hull_cloud;
  cur_state_ = cur_state;

  int rot_idx = -1;
  if (new_object)
  {
    // Reset boundary traversal data
    start_loc_arc_length_percent_ = 0.0;
    start_loc_push_sample_count_ = 0;
    start_loc_history_.clear();

    // NOTE: Initial start location is the dominant orientation
    // ROS_INFO_STREAM("Current state theta is: " << cur_state.x.theta);
    double min_angle_dist = FLT_MAX;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      double theta_i = atan2(hull_cloud.at(i).y - cur_state.x.y, hull_cloud.at(i).x - cur_state.x.x);
      double angle_dist_i = fabs(subPIAngle(theta_i - cur_state.x.theta));
      if (angle_dist_i < min_angle_dist)
      {
        min_angle_dist = angle_dist_i;
        rot_idx = i;
      }
    }
  }
  else
  {
    // Increment boundary location if necessary
    if (start_loc_history_.size() % num_start_loc_pushes_per_sample == 0)
    {
      start_loc_arc_length_percent_ += 1.0/num_start_loc_sample_locs;
      // ROS_INFO_STREAM("Incrementing arc length percent based on: " << num_start_loc_pushes_per_sample);
    }

    // Get initial object boundary location in the current world frame
    // ROS_INFO_STREAM("init_obj_point: " << start_loc_history_[0].boundary_loc_);
    pcl::PointXYZ init_loc_point = objectPointInWorldFrame(start_loc_history_[0].boundary_loc_, cur_state);
    // ROS_INFO_STREAM("init_loc_point: " << init_loc_point);

    // Find index of closest point on current boundary to the initial pushing location
    double min_dist = FLT_MAX;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      double dist_i = sqrDist(init_loc_point, hull_cloud.at(i));
      if (dist_i < min_dist)
      {
        min_dist = dist_i;
        rot_idx = i;
      }
    }
  }
  // Test hull_cloud orientation, reverse iteration if it is negative
  double pt0_theta = atan2(hull_cloud[rot_idx].y - cur_state.x.y, hull_cloud[rot_idx].x - cur_state.x.x);
  int pt1_idx = (rot_idx+1) % hull_cloud.size();
  double pt1_theta = atan2(hull_cloud[pt1_idx].y - cur_state.x.y, hull_cloud[pt1_idx].x - cur_state.x.x);
  bool reverse_data = false;
  if (subPIAngle(pt1_theta - pt0_theta) < 0)
  {
    reverse_data = true;
    // ROS_INFO_STREAM("Reversing data for boundaries");
  }

  // Compute cumulative distance around the boundary at each point
  std::vector<double> boundary_dists(hull_cloud.size(), 0.0);
  double boundary_length = 0.0;
  // ROS_INFO_STREAM("rot_idx is " << rot_idx);
  for (int i = 1; i <= hull_cloud.size(); ++i)
  {
    int idx0 = (rot_idx+i-1) % hull_cloud.size();
    int idx1 = (rot_idx+i) % hull_cloud.size();
    if (reverse_data)
    {
      idx0 = (hull_cloud.size()+rot_idx-i+1) % hull_cloud.size();
      idx1 = (hull_cloud.size()+rot_idx-i) % hull_cloud.size();
    }
    // NOTE: This makes boundary_dists[rot_idx] = 0.0, and we have no location at 100% the boundary_length
    boundary_dists[idx0] = boundary_length;
    double loc_dist = dist(hull_cloud[idx0], hull_cloud[idx1]);
    boundary_length += loc_dist;
  }

  // Find location at start_loc_arc_length_percent_ around the boundary
  double desired_boundary_dist = start_loc_arc_length_percent_*boundary_length;
  // ROS_INFO_STREAM("Finding location at dist " << desired_boundary_dist << " ~= " << start_loc_arc_length_percent_*100 << "\% of " << boundary_length);
  int boundary_loc_idx;
  double min_boundary_dist_diff = FLT_MAX;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double boundary_dist_diff_i = fabs(desired_boundary_dist - boundary_dists[i]);
    if (boundary_dist_diff_i < min_boundary_dist_diff)
    {
      min_boundary_dist_diff = boundary_dist_diff_i;
      boundary_loc_idx = i;
    }
  }

  // Get descriptor at the chosen location
  // ShapeLocations locs = tabletop_pushing::extractShapeContextFromSamples(hull_cloud, cur_obj, true);
  double gripper_spread = 0.05;
  pcl::PointXYZ boundary_loc = hull_cloud[boundary_loc_idx];
#ifdef USE_HKS_DESCRIPTOR
  ShapeDescriptor sd = tabletop_pushing::extractHKSAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                          boundary_loc, boundary_loc_idx,
                                                                          gripper_spread, hull_alpha,
                                                                          point_cloud_hist_res_);
#else // USE_HKS_DESCRIPTOR
  ShapeDescriptor sd = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                            boundary_loc, boundary_loc_idx,
                                                                            gripper_spread, hull_alpha,
                                                                            point_cloud_hist_res_);
#endif // USE_HKS_DESCRIPTOR
  // Add into pushing history in object frame
  ShapeLocation s_obj(worldPointInObjectFrame(boundary_loc, cur_state), sd);
  start_loc_history_.push_back(s_obj);
  ShapeLocation s_world(boundary_loc, sd);
  return s_world;
}

ShapeLocation chooseFixedGoalPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state,
                                          pcl::PointXYZ start_pt)

{
  double hull_alpha = 0.01;
  XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);
  hull_cloud_ = hull_cloud;
  cur_state_ = cur_state;
  double min_dist = FLT_MAX;
  int min_dist_idx = 0;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double pt_dist = dist(hull_cloud[i], start_pt);
    if (pt_dist < min_dist)
    {
      min_dist = pt_dist;
      min_dist_idx = i;
    }
  }
  double gripper_spread = 0.05;
  pcl::PointXYZ boundary_loc = hull_cloud[min_dist_idx];
#ifdef USE_HKS_DESCRIPTOR
  // cv::Mat boundary = visualizeObjectBoundarySamples(hull_cloud, cur_state);
  // cv::imshow("Obj boundary", boundary);
  ShapeDescriptor sd = tabletop_pushing::extractHKSAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                          boundary_loc, min_dist_idx,
                                                                          gripper_spread, hull_alpha,
                                                                          point_cloud_hist_res_);
#else // USE_HKS_DESCRIPTOR
  ShapeDescriptor sd = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                            boundary_loc, min_dist_idx,
                                                                            gripper_spread,
                                                                            hull_alpha, point_cloud_hist_res_);
#endif // USE_HKS_DESCRIPTOR
  ShapeLocation s_obj(worldPointInObjectFrame(boundary_loc, cur_state), sd);
  start_loc_history_.push_back(s_obj);

  ShapeLocation s_world(boundary_loc, sd);
  return s_world;
}

ShapeLocation chooseLearnedPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, std::string param_path,
                                        float& chosen_score)
{
  // Get features for all of the boundary locations
  float hull_alpha = 0.01;
  XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);
  float gripper_spread = 0.05;
  ShapeDescriptors sds = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                              gripper_spread, hull_alpha,
                                                                              point_cloud_hist_res_);
  // Set parameters for prediction
  // svm_parameter push_parameters;
  // push_parameters.svm_type = EPSILON_SVR;
  // push_parameters.kernel_type = PRECOMPUTED;
  // push_parameters.C = 2.0; // NOTE: only needed for training
  // push_parameters.p = 0.3; // NOTE: only needed for training
  // push_model.param = push_parameters;

  // TODO: Read in model SVs and coefficients
  ROS_INFO_STREAM("reading svm model: " << param_path);
  svm_model* push_model;
  push_model = svm_load_model(param_path.c_str());
  ROS_INFO_STREAM("read svm model: " << param_path);
  ROS_INFO_STREAM("svm_parameters.svm_type: " << push_model->param.svm_type);
  ROS_INFO_STREAM("svm_parameters.kernel_type: " << push_model->param.kernel_type);
  ROS_INFO_STREAM("number SVs: " << push_model->l);

  std::vector<double> pred_push_scores;
  chosen_score = FLT_MAX;
  int best_idx = -1;
  // Perform prediction at all sample locations
  for (int i = 0; i < sds.size(); ++i)
  {
    // ROS_INFO_STREAM("Predicting score for location " << i);
    // Set the data vector in libsvm format
    svm_node* x = new svm_node[sds[i].size()];
    for (int j = 0; j < sds[i].size(); ++j)
    {
      x[j].index = (j+1); // NOTE: 1 based indices
      x[j].value = sds[i][j];
    }
    // ROS_INFO_STREAM("Created svm_node vector of size: " << sds[i].size());
    // Perform prediction and convert out of log spacex
    double pred_log_score = svm_predict(push_model, x);
    double pred_score = exp(pred_log_score);
    // ROS_INFO_STREAM("Predicted score for location " << i << " of " << pred_score << " from log score " << pred_log_score);
    // Track the best score to know the location to return
    if (pred_score < chosen_score)
    {
      chosen_score = pred_score;
      best_idx = i;
    }
    pred_push_scores.push_back(pred_score);
    delete x;
  }
  ROS_INFO_STREAM("Chose best push location " << best_idx << " with score " << chosen_score);
  ROS_INFO_STREAM("Push location 3D: " << hull_cloud[best_idx]);
  // Return the location of the best score
  ShapeLocation loc;
  if (best_idx >= 0)
  {
    loc.boundary_loc_ = hull_cloud[best_idx];
    loc.descriptor_ = sds[best_idx];
  }
  return loc;
}

ShapeDescriptor getTrialDescriptor(std::string cloud_path, pcl::PointXYZ init_loc, double init_theta, bool new_object)
{
  int num_start_loc_pushes_per_sample = 3;
  int num_start_loc_sample_locs = 16;

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  ProtoObject cur_obj;
  //.cloud = ; // TODO: COPY from read in one?
  PushTrackerState cur_state;
  cur_state.x.x = init_loc.x;
  cur_state.x.y = init_loc.y;
  cur_state.x.theta = init_theta;
  cur_state.z = init_loc.z;
  cur_obj.centroid[0] = cur_state.x.x;
  cur_obj.centroid[1] = cur_state.x.y;
  cur_obj.centroid[2] = cur_state.z;
  // ROS_INFO_STREAM("Getting cloud: " << cloud_path);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_path, cur_obj.cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  // ROS_INFO_STREAM("Got cloud: " << cloud_path);
  ShapeLocation sl = chooseFixedGoalPushStartLoc(cur_obj, cur_state, new_object, num_start_loc_pushes_per_sample,
                                                 num_start_loc_sample_locs);
  return sl.descriptor_;
}

ShapeDescriptor getTrialDescriptor(std::string cloud_path, pcl::PointXYZ init_loc, double init_theta,
                                   pcl::PointXYZ start_pt)
{
  int num_start_loc_pushes_per_sample = 3;
  int num_start_loc_sample_locs = 16;

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  ProtoObject cur_obj;
  //.cloud = ; // TODO: COPY from read in one?
  PushTrackerState cur_state;
  cur_state.x.x = init_loc.x;
  cur_state.x.y = init_loc.y;
  cur_state.x.theta = init_theta;
  cur_state.z = init_loc.z;
  cur_obj.centroid[0] = cur_state.x.x;
  cur_obj.centroid[1] = cur_state.x.y;
  cur_obj.centroid[2] = cur_state.z;
  // ROS_INFO_STREAM("getting cloud: " << cloud_path);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_path, cur_obj.cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  // ROS_INFO_STREAM("Got cloud: " << cloud_path);
  ShapeLocation sl = chooseFixedGoalPushStartLoc(cur_obj, cur_state, start_pt);
  return sl.descriptor_;
}

ShapeLocation predictPushLocation(std::string cloud_path, pcl::PointXYZ init_loc, double init_theta,
                                  pcl::PointXYZ start_pt, std::string param_path)
{
  int num_start_loc_pushes_per_sample = 3;
  int num_start_loc_sample_locs = 16;

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  ProtoObject cur_obj;
  //.cloud = ; // TODO: COPY from read in one?
  PushTrackerState cur_state;
  cur_state.x.x = init_loc.x;
  cur_state.x.y = init_loc.y;
  cur_state.x.theta = init_theta;
  cur_state.z = init_loc.z;
  cur_obj.centroid[0] = cur_state.x.x;
  cur_obj.centroid[1] = cur_state.x.y;
  cur_obj.centroid[2] = cur_state.z;
  // ROS_INFO_STREAM("Getting cloud: " << cloud_path);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_path, cur_obj.cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  // ROS_INFO_STREAM("Got cloud: " << cloud_path);
  float chosen_score;
  ShapeLocation sl = chooseLearnedPushStartLoc(cur_obj, cur_state, param_path, chosen_score);
  return sl;
}

class TrialStuff
{
 public:
  TrialStuff(double init_x_, double init_y_, double init_z_, double init_theta_, std::string trial_id_, bool new_object_,
             double push_x_, double push_y_, double push_z_) :
      init_loc(init_x_, init_y_, init_z_), init_theta(init_theta_), trial_id(trial_id_), new_object(new_object_),
      start_pt(push_x_, push_y_, push_z_)
  {
  }
  pcl::PointXYZ init_loc;
  double init_theta;
  std::string trial_id;
  bool new_object;
  pcl::PointXYZ start_pt;
};

std::vector<TrialStuff> getTrialsFromFile(std::string aff_file_name)
{
  std::vector<TrialStuff> trials;
  std::ifstream trials_in(aff_file_name.c_str());

  bool next_line_trial = false;
  bool trial_is_start = true;
  int line_count = 0;
  int object_comment = 0;
  int trial_starts = 0;
  int bad_stops = 0;
  int good_stops = 0;
  int control_headers = 0;
  bool new_object = true;
  while(trials_in.good())
  {
    char c_line[4096];
    trials_in.getline(c_line, 4096);
    line_count++;
    if (next_line_trial)
    {
      // ROS_INFO_STREAM("Parsing trial_line: ");
      next_line_trial = false;

      // TODO: Parse this line!
      std::stringstream trial_line;
      trial_line << c_line;
      char trial_id_c[4096];
      trial_line.getline(trial_id_c, 4096, ' ');
      std::stringstream trial_id;
      trial_id << trial_id_c;
      // ROS_INFO_STREAM("Read trial_id: " << trial_id.str());
      double init_x, init_y, init_z, init_theta;
      trial_line >> init_x >> init_y >> init_z >> init_theta;
      double final_x, final_y, final_z, final_theta;
      trial_line >> final_x >> final_y >> final_z >> final_theta;
      double goal_x, goal_y, goal_theta;
      trial_line >> goal_x >> goal_y >> goal_theta;
      double push_start_x, push_start_y, push_start_z, push_start_theta;
      trial_line >> push_start_x >> push_start_y >> push_start_z;

      // ROS_INFO_STREAM("Init pose (" << init_x << ", " << init_y << ", " << init_z << ", " << init_theta << ")");
      // TODO: Read in start_point?!?
      new_object = !trials.size();
      TrialStuff trial(init_x, init_y, init_z, init_theta, trial_id.str(), new_object,
                       push_start_x, push_start_y, push_start_z);
      trials.push_back(trial);
    }
    if (c_line[0] == '#')
    {
      if (c_line[2] == 'o')
      {
        object_comment++;
        if (trial_is_start)
        {
          next_line_trial = true;
          trial_starts += 1;
          // ROS_INFO_STREAM("Read in start line");
        }
        else
        {
          // ROS_INFO_STREAM("Read in end line");
          good_stops++;
        }
        // Switch state
        trial_is_start = !trial_is_start;
      }
      else if (c_line[2] == 'x')
      {
        control_headers += 1;
      }
      else if (c_line[1] == 'B')
      {
        // ROS_WARN_STREAM("Read in bad line" << c_line);
        trial_is_start = true;
        trials.pop_back();
        bad_stops += 1;
      }
    }
  }
  trials_in.close();

  ROS_INFO_STREAM("Read in: " << line_count << " lines");
  ROS_INFO_STREAM("Read in: " << control_headers << " control headers");
  ROS_INFO_STREAM("Read in: " << object_comment << " trial headers");
  ROS_INFO_STREAM("Classified: " << trial_starts << " as starts");
  ROS_INFO_STREAM("Classified: " << bad_stops << " as bad");
  ROS_INFO_STREAM("Classified: " << good_stops << " as good");
  ROS_INFO_STREAM("Read in: " << trials.size() << " trials");
  return trials;
}

void writeNewFile(std::string out_file_name, std::vector<TrialStuff> trials, ShapeDescriptors descriptors)
{
  std::ofstream out_file(out_file_name.c_str());
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    for (unsigned int j = 0; j < descriptors[i].size(); ++j)
    {
      out_file << descriptors[i][j] << " ";
    }
    out_file << "\n";
  }
  out_file.close();
}

void writeNewExampleFile(std::string out_file_name, std::vector<TrialStuff> trials, ShapeDescriptors descriptors,
                         std::vector<double>& push_scores)
{
  ROS_INFO_STREAM("Writing example file: " << out_file_name << "\n");
  std::ofstream out_file(out_file_name.c_str());
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    out_file << push_scores[i] << " ";
    for (unsigned int j = 0; j < descriptors[i].size(); ++j)
    {
      if (descriptors[i][j] != 0.0)
      {
        out_file << (j+1) << ":" << descriptors[i][j] << " ";
      }
    }
    out_file << "\n";
  }
  out_file.close();
}

std::vector<double> readScoreFile(std::string file_path)
{
  std::ifstream data_in(file_path.c_str());
  std::vector<double> scores;
  while (data_in.good())
  {
    char c_line[4096];
    data_in.getline(c_line, 4096);
    std::stringstream line;
    line << c_line;
    double y;
    line >> y;
    if (!data_in.eof())
    {
      scores.push_back(y);
    }
  }
  data_in.close();
  return scores;
}

void drawScores(std::vector<double>& push_scores, std::string out_file_path)
{
  double max_y = 0.3;
  double min_y = -0.3;
  double max_x = 0.3;
  double min_x = -0.3;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

  for (int i = 0; i < hull_cloud_.size(); ++i)
  {
    pcl::PointXYZ obj_pt =  worldPointInObjectFrame(hull_cloud_[i], cur_state_);
    int img_x = objLocToIdx(obj_pt.x, min_x, max_x);
    int img_y = objLocToIdx(obj_pt.y, min_y, max_y);
    cv::Scalar color(128, 0, 0);
    cv::circle(footprint, cv::Point(img_x, img_y), 1, color, 3);
  }
  // HACK: Normalize across object classes
  double max_score = 0.0602445;
  for (int i = 0; i < start_loc_history_.size(); ++i)
  {
    int x = objLocToIdx(start_loc_history_[i].boundary_loc_.x, min_x, max_x);
    int y = objLocToIdx(start_loc_history_[i].boundary_loc_.y, min_y, max_y);
    // double score = -log(push_scores[i])/10;
    double score = push_scores[i]/M_PI;
    cv::Scalar color(0, score*255, (1-score)*255);
    // color[0] = 0.5;
    // color[1] = score;
    // color[2] = 0.5;
    // footprint.at<cv::Vec3f>(r,c) = color;
    cv::circle(footprint, cv::Point(x,y), 1, color, 3);
    cv::circle(footprint, cv::Point(x,y), 2, color, 3);
    cv::circle(footprint, cv::Point(x,y), 3, color, 3);
  }
  // ROS_INFO_STREAM("Max score is: " << max_score);
  ROS_INFO_STREAM("Writing image: " << out_file_path);
  cv::imwrite(out_file_path, footprint);

  cv::imshow("Push score", footprint);
  cv::waitKey();
}

pcl::PointXYZ pointClosestToAngle(double major_angle, XYZPointCloud& hull_cloud, Eigen::Vector4f centroid)
{
  pcl::PointXYZ closest;
  double min_angle_dist = FLT_MAX;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double angle_dist = fabs(major_angle - atan2(hull_cloud[i].y - centroid[1], hull_cloud[i].x - centroid[0]));
    if (angle_dist < min_angle_dist)
    {
      closest = hull_cloud[i];
      min_angle_dist = angle_dist;
    }
  }
  ROS_INFO_STREAM("Closest point " << closest << " at angle dist " << min_angle_dist);
  return closest;
}

void getMajorMinorBoundaryDists(std::string cloud_path, pcl::PointXYZ init_loc, pcl::PointXYZ start_pt,
                                double& major_dist, double& minor_dist)
{
  ProtoObject cur_obj;
  cur_obj.centroid[0] = init_loc.x;
  cur_obj.centroid[1] = init_loc.y;
  cur_obj.centroid[2] = init_loc.z;
  // ROS_INFO_STREAM("Getting cloud: " << cloud_path);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (cloud_path, cur_obj.cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  // ROS_INFO_STREAM("Got cloud: " << cloud_path);

  float hull_alpha = 0.01;
  XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);
  hull_cloud_ = hull_cloud;
  double min_dist = FLT_MAX;
  int min_dist_idx = 0;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double pt_dist = dist(hull_cloud[i], start_pt);
    if (pt_dist < min_dist)
    {
      min_dist = pt_dist;
      min_dist_idx = i;
    }
  }
  float gripper_spread = 0.05;
  pcl::PointXYZ boundary_loc = hull_cloud[min_dist_idx];

  // TODO: Get major/minor axis of hull_cloud
  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(hull_cloud.makeShared());
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
  Eigen::Vector4f centroid = pca.getMean();
  double minor_angle = atan2(eigen_vectors(1,0), eigen_vectors(0,0));
  double major_angle = minor_angle-0.5*M_PI;

  // TODO: Figure out points a and b of major axis intersection
  pcl::PointXYZ a = pointClosestToAngle(major_angle, hull_cloud, centroid);
  pcl::PointXYZ b = pointClosestToAngle(subPIAngle(major_angle+M_PI), hull_cloud, centroid);
  major_dist = std::min(sqrDistXY(boundary_loc, a), sqrDistXY(boundary_loc, b));
  // TODO: Figure out points c and d of minor axis intersection
  pcl::PointXYZ c  = pointClosestToAngle(minor_angle, hull_cloud, centroid);
  pcl::PointXYZ d = pointClosestToAngle(subPIAngle(minor_angle+M_PI), hull_cloud, centroid);
  minor_dist = std::min(sqrDistXY(boundary_loc, c), sqrDistXY(boundary_loc, d));
}

int main(int argc, char** argv)
{
  int seed = time(NULL);
  srand(seed);

  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);
  std::vector<TrialStuff> trials = getTrialsFromFile(aff_file_path);
  std::string score_file = "";
  bool draw_scores = argc > 4;
  ROS_INFO_STREAM("trials.size(): " << trials.size());
  std::vector<double> push_scores;
  if (draw_scores)
  {
    score_file = argv[4];
    ROS_INFO_STREAM("Reading score file " << score_file);
    push_scores = readScoreFile(score_file);
    ROS_INFO_STREAM("scores.size(): " << push_scores.size());
  }

  bool test_straw_man = false;
  // std::ofstream straw_scores_stream("/home/thermans/Desktop/straw_scores.txt");
  double max_score = -100.0;
  ShapeDescriptors descriptors;
  for (unsigned int i = 0; i < trials.size(); ++i)
  {
    std::string trial_id = trials[i].trial_id;
    pcl::PointXYZ init_loc = trials[i].init_loc;
    double init_theta = trials[i].init_theta;
    bool new_object = trials[i].new_object;
    // ROS_INFO_STREAM("trial_id: " << trial_id);
    // ROS_INFO_STREAM("init_loc: " << init_loc);
    // ROS_INFO_STREAM("init_theta: " << init_theta);
    // ROS_INFO_STREAM("new object: " << new_object);
    // ROS_INFO_STREAM("start_pt: " << trials[i].start_pt);
    std::stringstream cloud_path;
    cloud_path << data_directory_path << trial_id << "_obj_cloud.pcd";

    ShapeDescriptor sd = getTrialDescriptor(cloud_path.str(), init_loc, init_theta, trials[i].start_pt);
    // std::string param_path = "/home/thermans/Dropbox/Data/start_loc_learning/point_push/examples_line_dist/push_svm_1.model";
    // ShapeLocation chosen = predictPushLocation(cloud_path.str(), init_loc, init_theta, trials[i].start_pt, param_path);

    // if (draw_scores)
    // {
    //   // TODO: Get the image, draw the shape context, highlight score color
    //   // TODO: Get projection matrix
    //   std::stringstream hull_img_path;
    //   hull_img_path << data_directory_path << trial_id << "_obj_hull_disp.png";
    //   cv::Mat disp_img;
    //   ROS_INFO_STREAM("Reading image: " << hull_img_path.str());
    //   disp_img = cv::imread(hull_img_path.str());
    //   ROS_INFO_STREAM("Read image: " << hull_img_path.str());
    //   // double score = -log(push_scores[i])/10;
    //   double score = push_scores[i]/M_PI;
    //   cv::Vec3b score_color(0, score*255, (1-score)*255);
    //   for (int r = 0; r < disp_img.rows; ++r)
    //   {
    //     for (int c = 0; c < disp_img.cols; ++c)
    //     {
    //       if (disp_img.at<cv::Vec3b>(r,c)[0] == 0 && disp_img.at<cv::Vec3b>(r,c)[1] == 0 &&
    //           disp_img.at<cv::Vec3b>(r,c)[2] == 255)
    //       {
    //         disp_img.at<cv::Vec3b>(r,c) = score_color;
    //       }
    //       else if (disp_img.at<cv::Vec3b>(r,c)[0] == 0 && disp_img.at<cv::Vec3b>(r,c)[1] == 255 &&
    //                disp_img.at<cv::Vec3b>(r,c)[2] == 0)
    //       {
    //         disp_img.at<cv::Vec3b>(r,c) = cv::Vec3b(255,0,0);
    //       }

    //     }
    //   }
    //   ROS_INFO_STREAM("Score is " << push_scores[i] << "\n");
    //   cv::imshow("hull", disp_img);
    //   // cv::waitKey();
    //   if (push_scores[i] > max_score)
    //   {
    //     max_score = push_scores[i];
    //   }
    // }
    descriptors.push_back(sd);
  }
  // Feature testing below
  // ROS_INFO_STREAM("Constructing features matrices for x^2 kernel");
  // int local_hist_width = 6;
  // int local_hist_size = local_hist_width*local_hist_width;
  // int global_hist_size = 60;
  // cv::Mat local_feats(cv::Size(local_hist_size, descriptors.size()), CV_64FC1, cv::Scalar(0.0));
  // cv::Mat global_feats(cv::Size(global_hist_size, descriptors.size()), CV_64FC1, cv::Scalar(0.0));
  // std::vector<std::vector<double> > local_feats;
  // std::vector<std::vector<double> > global_feats;
  // cv::Mat global_feats(cv::Size(global_hist_size, descriptors.size()), CV_64FC1, cv::Scalar(0.0));
  // ROS_INFO_STREAM("feat_length: " << descriptors[0].size());
  // for (int r = 0; r < descriptors.size(); ++r)
  // {
  //   std::vector<double> local_row;
  //   local_feats.push_back(local_row);
  //   std::vector<double> global_row;
  //   global_feats.push_back(global_row);
  //   for (int c = 0; c < local_hist_size; ++c)
  //   {
  //     local_feats[r].push_back(descriptors[r][c]);
  //   }
  //   for (int c = local_hist_size; c < local_hist_size+global_hist_size; ++c)
  //   {
  //     global_feats[r].push_back(descriptors[r][c]);
  //   }
  // }
  // ROS_INFO_STREAM("Computing x^2 for " << descriptors.size() << " descriptors");
  // ROS_INFO_STREAM("Global_feats.size() (" << global_feats.size() << ", " << global_feats[0].size() << ")");
  // ROS_INFO_STREAM("Local_feats.size() (" << local_feats.size() << ", " << local_feats[0].size() << ")");
  // std::vector<std::vector<double> > K_global = chiSquareKernelBatch(global_feats, global_feats, 2.0);
  // std::vector<std::vector<double> > K_local = chiSquareKernelBatch(local_feats, local_feats, 2.5);
  // std::stringstream global_out;
  // for (int r = 0; r < K_global.size(); ++r)
  // {
  //   for (int c = 0; c < K_global[r].size(); ++c)
  //   {
  //     global_out << " " << K_global[r][c];
  //   }
  //   global_out << "\n";
  // }
  // ROS_INFO_STREAM("Global: \n" << global_out.str());
  // std::stringstream local_out;
  // for (int r = 0; r < K_local.size(); ++r)
  // {
  //   for (int c = 0; c < K_local[r].size(); ++c)
  //   {
  //     local_out << " " << K_local[r][c];
  //   }
  //   local_out << "\n";
  // }
  // ROS_INFO_STREAM("Local: \n" << local_out.str());
  // std::stringstream line_out;
  // for (int c = 0; c < local_feats[0].size(); ++c)
  // {
  //   for (int r = 0; r < local_feats.size(); ++r)
  //   {
  //     if (local_feats[r][c] > 0)
  //     {
  //       line_out << "\t(" << (r+1) << ", " << (c+1) << ")\t" << local_feats[r][c] << "\n";
  //     }
  //   }
  // }
  // ROS_INFO_STREAM("feat: " << line_out.str());

    // if (test_straw_man)
    // {
    //   double major_dist_i = 0.0;
    //   double minor_dist_i = 0.0;
    //   getMajorMinorBoundaryDists(cloud_path.str(), init_loc, trials[i].start_pt, major_dist_i, minor_dist_i);
    //   straw_scores_stream << push_scores[i] << " " << major_dist_i << " " << minor_dist_i << "\n";
    // }

  // std::stringstream out_file;
  writeNewExampleFile(out_file_path, trials, descriptors, push_scores);
  if (draw_scores)
  {
    // TODO: Pass in info to write these to disk again?
    // drawScores(push_scores, out_file_path);
    // drawScores(push_scores);
  }
  return 0;
}
