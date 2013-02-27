#ifndef shape_features_h_DEFINED
#define shape_features_h_DEFINED

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <cpl_visual_features/features/shape_context.h>
#include <geometry_msgs/PointStamped.h>
#include <vector>
#include <tabletop_pushing/point_cloud_segmentation.h>

namespace tabletop_pushing
{

class ShapeLocation
{
 public:
  ShapeLocation(geometry_msgs::Point boundary_loc, cpl_visual_features::ShapeDescriptor descriptor):
      boundary_loc_(boundary_loc), descriptor_(descriptor)
  {
  }
  geometry_msgs::Point boundary_loc_;
  cpl_visual_features::ShapeDescriptor descriptor_;
 private:
  ShapeLocation()
  {
  }
};

typedef std::vector<ShapeLocation> ShapeLocations;

cv::Mat getObjectFootprint(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud);

std::vector<cv::Point2f> getObjectBoundarySamples(ProtoObject& cur_obj);

ShapeLocations extractObjectShapeFeatures(ProtoObject& cur_obj);

cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs);

double shapeFeatureSimilarity(ShapeLocation& a, ShapeLocation& b);

};
#endif // shape_features_h_DEFINED
