/*********************************************************************
 * Software License Agreement (BSD License)
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
#include <ros/ros.h>

// TF
#include <tf/transform_datatypes.h>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl16/common/common.h>
#include <pcl16/common/eigen.h>
#include <pcl16/common/centroid.h>
#include <pcl16/ModelCoefficients.h>
#include <pcl16/sample_consensus/method_types.h>
#include <pcl16/sample_consensus/model_types.h>
#include <pcl16/segmentation/sac_segmentation.h>
#include <pcl16/segmentation/extract_clusters.h>
#include <pcl16/segmentation/segment_differences.h>
#include <pcl16/segmentation/organized_multi_plane_segmentation.h>
#include <pcl16/search/search.h>
#include <pcl16/search/kdtree.h>
#include <pcl16/filters/voxel_grid.h>
#include <pcl16/filters/passthrough.h>
#include <pcl16/filters/extract_indices.h>
#include <pcl16/surface/concave_hull.h>
#include <pcl16/registration/icp.h>
#include <pcl16/features/integral_image_normal.h>
#include <pcl16/features/normal_3d.h>

// STL
#include <sstream>
// Local
#include <tabletop_pushing/point_cloud_segmentation.h>

// #define DISPLAY_CLOUD_DIFF 1
#define randf() static_cast<float>(rand())/RAND_MAX

typedef pcl16::search::KdTree<pcl16::PointXYZ>::Ptr KdTreePtr;
typedef pcl16::search::KdTree<pcl16::PointXYZ>::KdTreeFLANNPtr KdTreeFLANNPtr;
using pcl16::PointXYZ;

namespace tabletop_pushing
{

PointCloudSegmentation::PointCloudSegmentation(
    boost::shared_ptr<tf::TransformListener> tf) :
    tf_(tf)
{
  for (int i = 0; i < 200; ++i)
  {
    cv::Vec3f rand_color;
    rand_color[0] = randf();
    rand_color[1] = randf()*0.5;
    rand_color[2] = randf();
    colors_.push_back(rand_color);
  }
}

/**
 * Function to determine the table plane in a point cloud
 *
 * @param cloud The cloud with the table as dominant plane.
 *
 * @return The centroid of the points belonging to the table plane.
 */
Eigen::Vector4f PointCloudSegmentation::getTablePlane(
    XYZPointCloud& cloud, XYZPointCloud& objs_cloud, XYZPointCloud& plane_cloud,
    bool find_concave_hull)
{
  XYZPointCloud cloud_downsampled;
  if (use_voxel_down_)
  {
    pcl16::VoxelGrid<PointXYZ> downsample;
    downsample.setInputCloud(cloud.makeShared());
    downsample.setLeafSize(voxel_down_res_, voxel_down_res_, voxel_down_res_);
    downsample.filter(cloud_downsampled);
  }

  // Filter Cloud to not look for table planes on the ground
  XYZPointCloud cloud_z_filtered, cloud_filtered;
  pcl16::PassThrough<PointXYZ> z_pass;
  if (use_voxel_down_)
  {
    z_pass.setInputCloud(cloud_downsampled.makeShared());
  }
  else
  {
    z_pass.setInputCloud(cloud.makeShared());
  }
  z_pass.setFilterFieldName("z");
  z_pass.setFilterLimits(min_table_z_, max_table_z_);
  z_pass.filter(cloud_z_filtered);

  // Filter to be just in the range in front of the robot
  pcl16::PassThrough<PointXYZ> x_pass;
  x_pass.setInputCloud(cloud_z_filtered.makeShared());
  x_pass.setFilterFieldName("x");
  x_pass.setFilterLimits(min_workspace_x_, max_workspace_x_);
  x_pass.filter(cloud_filtered);

  // Segment the tabletop from the points using RANSAC plane fitting
  pcl16::ModelCoefficients coefficients;
  pcl16::PointIndices plane_inliers;

  // Create the segmentation object
  pcl16::SACSegmentation<PointXYZ> plane_seg;
  plane_seg.setOptimizeCoefficients(true);
  // plane_seg.setModelType(pcl16::SACMODEL_PLANE);
  plane_seg.setModelType(pcl16::SACMODEL_PERPENDICULAR_PLANE);
  plane_seg.setMethodType(pcl16::SAC_RANSAC);
  plane_seg.setDistanceThreshold(table_ransac_thresh_);
  plane_seg.setInputCloud(cloud_filtered.makeShared());
  Eigen::Vector3f z_axis(0.0, 0.0, 1.0);
  plane_seg.setAxis(z_axis);
  // plane_seg.setEpsAngle(table_ransac_angle_thresh_);
  plane_seg.segment(plane_inliers, coefficients);
  pcl16::copyPointCloud(cloud_filtered, plane_inliers, plane_cloud);

  // Extract the outliers from the point clouds
  pcl16::ExtractIndices<PointXYZ> extract;
  pcl16::PointIndices plane_outliers;
  extract.setInputCloud(cloud_filtered.makeShared());
  extract.setIndices(boost::make_shared<pcl16::PointIndices>(plane_inliers));
  extract.setNegative(true);
  extract.filter(objs_cloud);

  // Estimate hull from the inlier points
  if (find_concave_hull)
  {
    ROS_INFO_STREAM("finding concave hull. Plane size: " <<
                    plane_cloud.size());
    XYZPointCloud hull_cloud;
    pcl16::ConcaveHull<PointXYZ> hull;
    hull.setInputCloud(plane_cloud.makeShared());
    hull.setAlpha(hull_alpha_);
    hull.reconstruct(hull_cloud);
    ROS_INFO_STREAM("hull_cloud.size() " << hull_cloud.size());
    // TODO: Return the hull_cloud
    // TODO: Figure out if stuff is inside the hull
  }

  // Extract the plane members into their own point cloud
  Eigen::Vector4f table_centroid;
  pcl16::compute3DCentroid(plane_cloud, table_centroid);
  // cv::Size img_size(320, 240);
  // cv::Mat plane_img(img_size, CV_8UC1, cv::Scalar(0));
  // projectPointCloudIntoImage(plane_cloud, plane_img, cur_camera_header_.frame_id, 255);
  // cv::imshow("table plane", plane_img);

  return table_centroid;
}


/**
 * Function to segment independent spatial regions from a supporting plane
 *
 * @param input_cloud The point cloud to operate on.
 * @param extract_table True if the table plane should be extracted
 *
 * @return The object clusters.
 */
ProtoObjects PointCloudSegmentation::findTabletopObjects(XYZPointCloud& input_cloud, bool use_mps)
{
  XYZPointCloud objs_cloud;
  return findTabletopObjects(input_cloud, objs_cloud, use_mps);
}

/**
 * Function to segment independent spatial regions from a supporting plane
 *
 * @param input_cloud The point cloud to operate on.
 * @param objs_cloud  The point cloud containing the object points.
 * @param extract_table True if the table plane should be extracted
 *
 * @return The object clusters.
 */
ProtoObjects PointCloudSegmentation::findTabletopObjects(XYZPointCloud& input_cloud,
                                                         XYZPointCloud& objs_cloud, bool use_mps)
{
  XYZPointCloud table_cloud;
  return findTabletopObjects(input_cloud, objs_cloud, table_cloud, use_mps);
}

ProtoObjects PointCloudSegmentation::findTabletopObjects(XYZPointCloud& input_cloud,
                                                         XYZPointCloud& objs_cloud,
                                                         XYZPointCloud& plane_cloud, bool use_mps)
{
  if (use_mps)
  {
    return findTabletopObjectsMPS(input_cloud, objs_cloud, plane_cloud);
  }
  else
  {
    return findTabletopObjectsCluster(input_cloud, objs_cloud, plane_cloud);
  }

}

ProtoObjects PointCloudSegmentation::findTabletopObjectsMPS(XYZPointCloud& input_cloud,
                                                            XYZPointCloud& objs_cloud,
                                                            XYZPointCloud& plane_cloud)
{
  ROS_WARN_STREAM("Finding tabletop objects MPS!");
  pcl16::IntegralImageNormalEstimation<PointXYZ, pcl16::Normal> ne;
  ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor (0.03f);
  ne.setNormalSmoothingSize (20.0f);
  pcl16::PointCloud<pcl16::Normal>::Ptr normal_cloud (new pcl16::PointCloud<pcl16::Normal>);
  ne.setInputCloud(input_cloud.makeShared());
  ne.compute(*normal_cloud);

  cv::Mat normal_img(cv::Size(input_cloud.width, input_cloud.height), CV_32FC3, cv::Scalar(0));
  for (int x = 0; x < normal_img.cols; ++x)
    for (int y = 0; y < normal_img.rows; ++y)
  {
    cv::Vec3f norm;
    norm[0] = abs(normal_cloud->at(x,y).normal_x);
    norm[1] = abs(normal_cloud->at(x,y).normal_y);
    norm[2] = abs(normal_cloud->at(x,y).normal_z);
    normal_img.at<cv::Vec3f>(y,x) = norm;
  }
  cv::imshow("normals", normal_img);
  // cv::waitKey();

  pcl16::OrganizedMultiPlaneSegmentation<PointXYZ, pcl16::Normal, pcl16::Label> mps;
  // TODO: Check these parameters
  // TODO: Expose them in the launch file
  mps.setMinInliers(mps_min_inliers_);
  mps.setAngularThreshold(0.017453 *mps_min_angle_thresh_); // 2 degrees
  mps.setDistanceThreshold(mps_min_dist_thresh_); // 2cm
  mps.setInputNormals(normal_cloud);
  mps.setInputCloud(input_cloud.makeShared());
  std::vector<pcl16::PlanarRegion<PointXYZ>,
              Eigen::aligned_allocator<pcl16::PlanarRegion<PointXYZ> > > regions;
  std::vector<pcl16::ModelCoefficients> coefficients;
  std::vector<pcl16::PointIndices> point_indices;
  pcl16::OrganizedMultiPlaneSegmentation<PointXYZ, pcl16::Normal, pcl16::Label>::PointCloudLPtr labels;
  std::vector<pcl16::PointIndices> label_indices;
  std::vector<pcl16::PointIndices> boundary_indices;
  regions.clear();
  point_indices.clear();
  label_indices.clear();
  boundary_indices.clear();
  ROS_WARN_STREAM("Segmenting and refining!");
  mps.segmentAndRefine(regions);
  ROS_WARN_STREAM("Segmented and refined!");
  // mps.segmentAndRefine(regions, coefficients, point_indices, labels, label_indices, boundary_indices);

  // TODO: Get table plane
  // TODO: Create objects and their clouds
  // TODO: Filter out arm
  ProtoObjects objs;
  ROS_WARN_STREAM("Iterating throught " << regions.size() << " regions!");
  for (size_t i = 0; i < regions.size (); i++)
  {
    ProtoObject po;
    po.push_history.clear();
    po.boundary_angle_dist.clear();
    po.id = i;
    // TODO: Get object point cloud
    pcl16::copyPointCloud(input_cloud, point_indices[i], po.cloud);
    ROS_INFO_STREAM("Object has " << po.cloud.size() << " points");
    po.centroid[0] = regions[i].getCentroid()[0];
    po.centroid[1] = regions[i].getCentroid()[1];
    po.centroid[2] = regions[i].getCentroid()[2];
    po.centroid[3] = 1.0;
    po.moved = false;
    po.transform = Eigen::Matrix4f::Identity();
    po.singulated = false;
    objs.push_back(po);
    Eigen::Vector4f model = regions[i].getCoefficients ();
    // TODO: Save object counter
    XYZPointCloud boundary_cloud;
    boundary_cloud.points = regions[i].getContour ();
    printf ("Centroid: (%f, %f, %f)\n  Coefficients: (%f, %f, %f, %f)\n Inliers: %d\n",
            po.centroid[0], po.centroid[1], po.centroid[2],
            model[0], model[1], model[2], model[3],
            boundary_cloud.points.size ());
  }
  return objs;
}

/**
 * Function to segment independent spatial regions from a supporting plane
 *
 * @param input_cloud The point cloud to operate on.
 * @param objs_cloud  The point cloud containing the object points.
 * @param plane_cloud  The point cloud containing the table plane points.
 * @param extract_table True if the table plane should be extracted
 *
 * @return The object clusters.
 */
ProtoObjects PointCloudSegmentation::findTabletopObjectsCluster(XYZPointCloud& input_cloud,
                                                                XYZPointCloud& objs_cloud,
                                                                XYZPointCloud& plane_cloud)
{
  // Get table plane
  table_centroid_ = getTablePlane(input_cloud, objs_cloud, plane_cloud,
                                  false);
  min_workspace_z_ = table_centroid_[2];

  // ROS_INFO_STREAM("Estimating normals!");
  // NormalCloud::Ptr normal_cloud (new pcl16::PointCloud<pcl16::Normal>);
  // pcl16::IntegralImageNormalEstimation<PointXYZ, pcl16::Normal> ne;
  // ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  // ne.setMaxDepthChangeFactor(0.03f);
  // ne.setNormalSmoothingSize(20.0f);
  // ne.setInputCloud(input_cloud.makeShared());
  // ne.compute(*normal_cloud);

  XYZPointCloud objects_cloud_down = downsampleCloud(objs_cloud);
  // Find independent regions
  if (objects_cloud_down.size() < 1)
  {
    ProtoObjects objs;
    return objs;
  }
  ProtoObjects objs = clusterProtoObjects(objects_cloud_down);
  return objs;
}

/**
 * Function to segment point cloud regions using euclidean clustering
 *
 * @param objects_cloud The cloud of objects to cluster
 *
 * @return The independent clusters
 */
ProtoObjects PointCloudSegmentation::clusterProtoObjects(XYZPointCloud& objects_cloud)
{
  std::vector<pcl16::PointIndices> clusters;
  pcl16::EuclideanClusterExtraction<PointXYZ> pcl_cluster;
  const KdTreePtr clusters_tree(new pcl16::search::KdTree<PointXYZ>);
  clusters_tree->setInputCloud(objects_cloud.makeShared());

  pcl_cluster.setClusterTolerance(cluster_tolerance_);
  pcl_cluster.setMinClusterSize(min_cluster_size_);
  pcl_cluster.setMaxClusterSize(max_cluster_size_);
  pcl_cluster.setSearchMethod(clusters_tree);
  pcl_cluster.setInputCloud(objects_cloud.makeShared());
  pcl_cluster.extract(clusters);
  ROS_DEBUG_STREAM("Number of clusters found matching the given constraints: "
                   << clusters.size());

  ProtoObjects objs;
  for (unsigned int i = 0; i < clusters.size(); ++i)
  {
    // Create proto objects from the point cloud
    ProtoObject po;
    po.push_history.clear();
    po.boundary_angle_dist.clear();
    pcl16::copyPointCloud(objects_cloud, clusters[i], po.cloud);
    pcl16::compute3DCentroid(po.cloud, po.centroid);
    po.id = i;
    po.moved = false;
    po.transform = Eigen::Matrix4f::Identity();
    po.singulated = false;
    objs.push_back(po);
  }
  return objs;
}

/**
 * Perform Iterated Closest Point between two proto objects.
 *
 * @param a The first object
 * @param b The second object
 *
 * @return The ICP fitness score of the match
 */
double PointCloudSegmentation::ICPProtoObjects(ProtoObject& a, ProtoObject& b,
                                               Eigen::Matrix4f& transform)
{
  // TODO: Investigate this!
  // pcl16::IterativeClosestPointNonLinear<PointXYZ, PointXYZ> icp;
  pcl16::IterativeClosestPoint<PointXYZ, PointXYZ> icp;
  icp.setMaximumIterations(icp_max_iters_);
  icp.setTransformationEpsilon(icp_transform_eps_);
  icp.setMaxCorrespondenceDistance(icp_max_cor_dist_);
  icp.setRANSACOutlierRejectionThreshold(icp_ransac_thresh_);
  icp.setInputCloud(boost::make_shared<XYZPointCloud>(a.cloud));
  icp.setInputTarget(boost::make_shared<XYZPointCloud>(b.cloud));
  XYZPointCloud aligned;
  icp.align(aligned);
  double score = icp.getFitnessScore();
  transform = icp.getFinalTransformation();
  return score;
}

/**
 * Find the regions that have moved between two point clouds
 *
 * @param prev_cloud The first cloud to use in differencing
 * @param cur_cloud The second cloud to use
 *
 * @return The new set of objects that have moved in the second cloud
 */
ProtoObjects PointCloudSegmentation::getMovedRegions(XYZPointCloud& prev_cloud,
                                                     XYZPointCloud& cur_cloud,
                                                     std::string suf)
{
  // cloud_out = prev_cloud - cur_cloud
  pcl16::SegmentDifferences<PointXYZ> pcl_diff;
  pcl_diff.setDistanceThreshold(cloud_diff_thresh_);
  pcl_diff.setInputCloud(prev_cloud.makeShared());
  pcl_diff.setTargetCloud(cur_cloud.makeShared());
  XYZPointCloud cloud_out;
  pcl_diff.segment(cloud_out);
  if (cloud_out.size() < 1)
  {
    ROS_INFO_STREAM("Returning nothing moved as there are no points.");
    ProtoObjects moved;
    return moved;
  }

  ProtoObjects moved = clusterProtoObjects(cloud_out);

#ifdef DISPLAY_CLOUD_DIFF
  cv::Size img_size(320, 240);
  cv::Mat moved_img = projectProtoObjectsIntoImage(moved, img_size,
                                                   prev_cloud.header.frame_id);
  std::stringstream cluster_title;
  cluster_title << "moved clusters" << suf;
  displayObjectImage(moved_img, cluster_title.str());
#endif // DISPLAY_CLOUD_DIFF
  return moved;
}

/**
 * Match moved regions to previously extracted protoobjects
 *
 * @param objs The previously extracted objects
 * @param moved_regions The regions that have been detected as having moved
 *
 */
void PointCloudSegmentation::matchMovedRegions(ProtoObjects& objs,
                                               ProtoObjects& moved_regions)
{
  // Determining which previous objects have moved
  for (unsigned int i = 0; i < moved_regions.size(); ++i)
  {
    for (unsigned int j = 0; j < objs.size(); ++j)
    {
      if(cloudsIntersect(objs[j].cloud, moved_regions[i].cloud))
      {
        if (!objs[j].moved)
        {
          objs[j].moved = true;
        }
      }
    }
  }
}

pcl16::ModelCoefficients PointCloudSegmentation::fitCylinderRANSAC(ProtoObject& obj, XYZPointCloud& cylinder_cloud)
{
  pcl16::NormalEstimation<PointXYZ, pcl16::Normal> ne;
  ne.setInputCloud(obj.cloud.makeShared());
  pcl16::search::KdTree<PointXYZ>::Ptr tree (new pcl16::search::KdTree<PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (0.03);
  ne.compute(obj.normals);

  // cv::Mat normals_img(cv::Size(320,240), CV_32FC3, cv::Scalar(0.0,0.0,0.0));
  // for (int i = 0; i < obj.normals.size(); ++i)
  // {
  //   cv::Vec3f norm;
  //   norm[0] = abs(obj.normals[i].normal_x);
  //   norm[1] = abs(obj.normals[i].normal_y);
  //   norm[2] = abs(obj.normals[i].normal_z);
  //   cv::Point idx = projectPointIntoImage(obj.cloud[i], obj.cloud.header.frame_id,
  //                                         cur_camera_header_.frame_id);
  //   normals_img.at<cv::Vec3f>(idx.y,idx.x) = norm;
  // }
  // cv::imshow("normals", normals_img);

  // Create the segmentation object
  pcl16::ModelCoefficients coefficients;
  Eigen::Vector3f z_axis(0.0,0.0,1.0);
  pcl16::PointIndices cylinder_inliers;
  pcl16::SACSegmentationFromNormals<PointXYZ,pcl16::Normal> cylinder_seg;
  cylinder_seg.setOptimizeCoefficients(true);
  cylinder_seg.setModelType(pcl16::SACMODEL_CYLINDER);
  cylinder_seg.setMethodType(pcl16::SAC_RANSAC);
  cylinder_seg.setDistanceThreshold(cylinder_ransac_thresh_);
  cylinder_seg.setAxis(z_axis);
  // cylinder_seg.setEpsAngle(cylinder_ransac_angle_thresh_);
  cylinder_seg.setInputCloud(obj.cloud.makeShared());
  cylinder_seg.setInputNormals(obj.normals.makeShared());
  cylinder_seg.segment(cylinder_inliers, coefficients);

  pcl16::copyPointCloud(obj.cloud, cylinder_inliers, cylinder_cloud);
  return coefficients;
}

/**
 * Method to fit a sphere to a segmented object
 *
 * @param obj The segmented object we are modelling as a sphere
 * @param sphere_cloud The cloud resulting from the sphere fit
 *
 * @return The model of the sphere
 */
pcl16::ModelCoefficients PointCloudSegmentation::fitSphereRANSAC(ProtoObject& obj, XYZPointCloud& sphere_cloud)
{
  // Create the segmentation object
  pcl16::ModelCoefficients coefficients;
  pcl16::PointIndices sphere_inliers;
  pcl16::SACSegmentation<PointXYZ> sphere_seg;
  sphere_seg.setOptimizeCoefficients(true);
  sphere_seg.setModelType(pcl16::SACMODEL_SPHERE);
  sphere_seg.setMethodType(pcl16::SAC_RANSAC);
  sphere_seg.setDistanceThreshold(sphere_ransac_thresh_);
  sphere_seg.setInputCloud(obj.cloud.makeShared());
  sphere_seg.segment(sphere_inliers, coefficients);

  pcl16::copyPointCloud(obj.cloud, sphere_inliers, sphere_cloud);
  return coefficients;
}

/**
 * Naively determine if two point clouds intersect based on distance threshold
 * between points.
 *
 * @param cloud0 First cloud for interesection test
 * @param cloud1 Second cloud for interesection test
 *
 * @return true if any points from cloud0 and cloud1 have distance less than
 * voxel_down_res_
 */
bool PointCloudSegmentation::cloudsIntersect(XYZPointCloud cloud0,
                                             XYZPointCloud cloud1)
{
  int moved_count = 0;
  for (unsigned int i = 0; i < cloud0.size(); ++i)
  {
    const PointXYZ pt0 = cloud0.at(i);
    for (unsigned int j = 0; j < cloud1.size(); ++j)
    {
      const PointXYZ pt1 = cloud1.at(j);
      if (dist(pt0, pt1) < cloud_intersect_thresh_)
      {
        moved_count++;
      }
      if (moved_count > moved_count_thresh_)
      {
        return true;
      }
    }
  }
  return false;
}

bool PointCloudSegmentation::cloudsIntersect(XYZPointCloud cloud0,
                                             XYZPointCloud cloud1,
                                             double thresh)
{
  for (unsigned int i = 0; i < cloud0.size(); ++i)
  {
    const PointXYZ pt0 = cloud0.at(i);
    for (unsigned int j = 0; j < cloud1.size(); ++j)
    {
      const PointXYZ pt1 = cloud1.at(j);
      if (dist(pt0, pt1) < thresh) return true;
    }
  }
  return false;
}

bool PointCloudSegmentation::pointIntersectsCloud(XYZPointCloud cloud,
                                                  geometry_msgs::Point pt,
                                                  double thresh)
{
  for (unsigned int i = 0; i < cloud.size(); ++i)
  {
    const PointXYZ pt_c = cloud.at(i);
    if (dist(pt_c, pt) < thresh) return true;
  }
  return false;
}

float PointCloudSegmentation::pointLineXYDist(PointXYZ p,
                                              Eigen::Vector3f vec,
                                              Eigen::Vector4f base)
{
  Eigen::Vector3f x0(p.x,p.y,0.0);
  Eigen::Vector3f x1(base[0],base[1],0.0);
  Eigen::Vector3f x2 = x1+vec;
  Eigen::Vector3f num = (x0 - x1);
  num = num.cross(x0 - x2);
  Eigen::Vector3f den = x2 - x1;
  float d = num.norm()/den.norm();
  return d;
}

XYZPointCloud PointCloudSegmentation::lineCloudIntersection(
    XYZPointCloud& cloud, Eigen::Vector3f vec, Eigen::Vector4f base)
{
  // Define parametric model of the line defined by base and vec and
  // test cloud memebers for distance from the line, if the distance is less
  // than epsilon say it intersects and add to the output set.
  pcl16::PointIndices line_inliers;
  for (unsigned int i = 0; i < cloud.size(); ++i)
  {
    const PointXYZ pt = cloud.at(i);
    if (pointLineXYDist(pt, vec, base) < cloud_intersect_thresh_)
    {
      line_inliers.indices.push_back(i);
    }
  }

  // Extract the interesecting points of the line.
  XYZPointCloud line_cloud;
  pcl16::ExtractIndices<PointXYZ> extract;
  extract.setInputCloud(cloud.makeShared());
  extract.setIndices(boost::make_shared<pcl16::PointIndices>(line_inliers));
  extract.filter(line_cloud);
  return line_cloud;
}

std::vector<PointXYZ> PointCloudSegmentation::lineCloudIntersectionEndPoints(
    XYZPointCloud& cloud, Eigen::Vector3f vec, Eigen::Vector4f base)
{
  XYZPointCloud intersection = lineCloudIntersection(cloud, vec, base);
  unsigned int min_y_idx = intersection.size();
  unsigned int max_y_idx = intersection.size();
  unsigned int min_x_idx = intersection.size();
  unsigned int max_x_idx = intersection.size();
  float min_y = FLT_MAX;
  float max_y = -FLT_MAX;
  float min_x = FLT_MAX;
  float max_x = -FLT_MAX;
  for (unsigned int i = 0; i < intersection.size(); ++i)
  {
    if (intersection.at(i).y < min_y)
    {
      min_y = intersection.at(i).y;
      min_y_idx = i;
    }
    if (intersection.at(i).y > max_y)
    {
      max_y = intersection.at(i).y;
      max_y_idx = i;
    }
    if (intersection.at(i).x < min_x)
    {
      min_x = intersection.at(i).x;
      min_x_idx = i;
    }
    if (intersection.at(i).x > max_x)
    {
      max_x = intersection.at(i).x;
      max_x_idx = i;
    }
  }
  const double y_dist_obs = max_y - min_y;
  const double x_dist_obs = max_x - min_x;
  int start_idx = min_x_idx;
  int end_idx = max_x_idx;

  if (x_dist_obs > y_dist_obs)
  {
    // Use X index
    if (vec[0] > 0)
    {
      // Use min
      start_idx = min_x_idx;
      end_idx = max_x_idx;
    }
    else
    {
      // use max
      start_idx = max_x_idx;
      end_idx = min_x_idx;
    }
  }
  else
  {
    // Use Y index
    if (vec[1] > 0)
    {
      // Use min
      start_idx = min_y_idx;
      end_idx = max_y_idx;
    }
    else
    {
      // use max
      start_idx = max_y_idx;
      end_idx = min_y_idx;
    }
  }
  std::vector<PointXYZ> points;
  PointXYZ start_point, end_point;
  start_point.x = intersection.at(start_idx).x;
  start_point.y = intersection.at(start_idx).y;
  start_point.z = intersection.at(start_idx).z;
  end_point.x = intersection.at(end_idx).x;
  end_point.y = intersection.at(end_idx).y;
  end_point.z = intersection.at(end_idx).z;
  points.push_back(start_point);
  points.push_back(end_point);
  return points;
}

/**
 * Filter a point cloud to only be above the estimated table and within the
 * workspace in x, then downsample the voxels.
 *
 * @param cloud_in The cloud to filter and downsample
 *
 * @return The downsampled cloud
 */
XYZPointCloud PointCloudSegmentation::downsampleCloud(XYZPointCloud& cloud_in)
{
  XYZPointCloud cloud_z_filtered, cloud_x_filtered, cloud_down;
  pcl16::PassThrough<PointXYZ> z_pass;
  z_pass.setFilterFieldName("z");
  ROS_DEBUG_STREAM("Number of points in cloud_in is: " <<
                   cloud_in.size());
  z_pass.setInputCloud(cloud_in.makeShared());
  z_pass.setFilterLimits(min_workspace_z_, max_workspace_z_);
  z_pass.filter(cloud_z_filtered);
  ROS_DEBUG_STREAM("Number of points in cloud_z_filtered is: " <<
                   cloud_z_filtered.size());

  pcl16::PassThrough<PointXYZ> x_pass;
  x_pass.setInputCloud(cloud_z_filtered.makeShared());
  x_pass.setFilterFieldName("x");
  x_pass.setFilterLimits(min_workspace_x_, max_workspace_x_);
  x_pass.filter(cloud_x_filtered);

  pcl16::VoxelGrid<PointXYZ> downsample_outliers;
  downsample_outliers.setInputCloud(cloud_x_filtered.makeShared());
  downsample_outliers.setLeafSize(voxel_down_res_, voxel_down_res_,
                                  voxel_down_res_);
  downsample_outliers.filter(cloud_down);
  ROS_DEBUG_STREAM("Number of points in objs_downsampled: " <<
                   cloud_down.size());
  return cloud_down;
}

/**
 * Method to project the current proto objects into an image
 *
 * @param objs The set of objects
 * @param img_in An image of correct size for the projection
 * @param target_frame The frame of the associated image
 *
 * @return Image containing the projected objects
 */
cv::Mat PointCloudSegmentation::projectProtoObjectsIntoImage(ProtoObjects& objs,
                                                             cv::Size img_size,
                                                             std::string target_frame)
{
  cv::Mat obj_img(img_size, CV_8UC1, cv::Scalar(0));
  for (unsigned int i = 0; i < objs.size(); ++i)
  {
    projectPointCloudIntoImage(objs[i].cloud, obj_img,
                               cur_camera_header_.frame_id, i+1);
  }

  return obj_img;
}

/**
 * Visualization function of proto objects projected into an image
 *
 * @param obj_img The projected objects image
 * @param objs The set of proto objects
 */
cv::Mat PointCloudSegmentation::displayObjectImage(cv::Mat& obj_img,
                                                   std::string win_name,
                                                   bool use_display)
{
  cv::Mat obj_disp_img(obj_img.size(), CV_32FC3, cv::Scalar(0.0,0.0,0.0));
  for (int r = 0; r < obj_img.rows; ++r)
  {
    for (int c = 0; c < obj_img.cols; ++c)
    {
      unsigned int id = obj_img.at<uchar>(r,c);
      if (id > 0)
      {
        obj_disp_img.at<cv::Vec3f>(r,c) = colors_[id-1];
      }
    }
  }
  if (use_display)
  {
    cv::imshow(win_name, obj_disp_img);
  }
  return obj_disp_img;
}

void PointCloudSegmentation::projectPointCloudIntoImage(XYZPointCloud& cloud,
                                                        cv::Mat& lbl_img,
                                                        std::string target_frame,
                                                        unsigned int id)
{
  for (unsigned int i = 0; i < cloud.size(); ++i)
  {
    cv::Point img_idx = projectPointIntoImage(cloud.at(i),
                                              cloud.header.frame_id,
                                              target_frame);
    lbl_img.at<uchar>(img_idx.y, img_idx.x) = id;
  }
}

cv::Point PointCloudSegmentation::projectPointIntoImage(Eigen::Vector3f cur_point_eig,
                                                        std::string point_frame,
                                                        std::string target_frame)
{
  geometry_msgs::PointStamped cur_point;
  cur_point.header.frame_id = point_frame;
  cur_point.point.x = cur_point_eig[0];
  cur_point.point.y = cur_point_eig[1];
  cur_point.point.z = cur_point_eig[2];
  return projectPointIntoImage(cur_point, target_frame);
}

cv::Point PointCloudSegmentation::projectPointIntoImage(PointXYZ cur_point_pcl,
                                                        std::string point_frame,
                                                        std::string target_frame)
{
  geometry_msgs::PointStamped cur_point;
  cur_point.header.frame_id = point_frame;
  cur_point.point.x = cur_point_pcl.x;
  cur_point.point.y = cur_point_pcl.y;
  cur_point.point.z = cur_point_pcl.z;
  return projectPointIntoImage(cur_point, target_frame);
}

void PointCloudSegmentation::projectPointCloudIntoImage(XYZPointCloud& cloud,
                                                        cv::Mat& lbl_img)
{
  for (unsigned int i = 0; i < cloud.size(); ++i)
  {
    cv::Point img_idx = projectPointIntoImage(cloud.at(i),
                                              cloud.header.frame_id,
                                              cur_camera_header_.frame_id);
    lbl_img.at<uchar>(img_idx.y, img_idx.x) = 1;
  }
}

cv::Point PointCloudSegmentation::projectPointIntoImage(
    geometry_msgs::PointStamped cur_point)
{
  return projectPointIntoImage(cur_point, cur_camera_header_.frame_id);
}

cv::Point PointCloudSegmentation::projectPointIntoImage(
    geometry_msgs::PointStamped cur_point, std::string target_frame)
{
  cv::Point img_loc;
  try
  {
    // Transform point into the camera frame
    geometry_msgs::PointStamped image_frame_loc_m;
    tf_->transformPoint(target_frame, cur_point, image_frame_loc_m);

    // Project point onto the image
    img_loc.x = static_cast<int>((cam_info_.K[0]*image_frame_loc_m.point.x +
                                  cam_info_.K[2]*image_frame_loc_m.point.z) /
                                 image_frame_loc_m.point.z);
    img_loc.y = static_cast<int>((cam_info_.K[4]*image_frame_loc_m.point.y +
                                  cam_info_.K[5]*image_frame_loc_m.point.z) /
                                 image_frame_loc_m.point.z);

    // Downsample poses if the image is downsampled
    for (int i = 0; i < num_downsamples_; ++i)
    {
      img_loc.x /= 2;
      img_loc.y /= 2;
    }
  }
  catch (tf::TransformException e)
  {
    ROS_ERROR_STREAM("Error projecting 3D point into image plane.");
    ROS_ERROR_STREAM(e.what());
    ROS_ERROR_STREAM("cur point header is: " << cur_point.header.frame_id);
    ROS_ERROR_STREAM("target frame is: " << target_frame);
  }
  return img_loc;
}

};
