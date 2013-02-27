#include <cpl_visual_features/features/shape_context.h>
#include <cpl_visual_features/extern/lap_cpp/lap.h>
#include <math.h>
#include <string>
#include <iostream>

namespace cpl_visual_features
{

void swapImages(cv::Mat& a, cv::Mat& b)
{
  cv::Mat c = a;
  a = b;
  b = c;
}

void swapSamples(Samples& a, Samples& b)
{
  Samples c = a;
  a = b;
  b = c;
}

double compareShapes(cv::Mat& imageA, cv::Mat& imageB, double epsilonCost, bool write_images,
                     std::string filePath, int max_displacement, std::string filePostFix)
{
  cv::Mat edge_imageA(imageA.size(), imageA.type());
  cv::Mat edge_imageB(imageB.size(), imageB.type());
  cv::Mat edge_imageA_raw;
  cv::Mat edge_imageB_raw;

  // do edge detection
  cv::Canny(imageA, edge_imageA, 0.05, 0.5);
  cv::Canny(imageB, edge_imageB, 0.05, 0.5);
  edge_imageA.copyTo(edge_imageA_raw);
  edge_imageB.copyTo(edge_imageB_raw);

  // sample a subset of the edge pixels
  Samples samplesA = samplePoints(edge_imageA);
  Samples samplesB = samplePoints(edge_imageB);
  // std::cout << "samplesA.size() " << samplesA.size() << std::endl;
  // std::cout << "samplesB.size() " << samplesB.size() << std::endl;

  // Swap images if B is shorter than A;
  if (samplesA.size() > samplesB.size())
  {
    swapImages(imageA, imageB);
    swapImages(edge_imageA, edge_imageB);
    swapImages(edge_imageA_raw, edge_imageB_raw);
    swapSamples(samplesA, samplesB);
  }

  // construct shape descriptors for each sample
  ShapeDescriptors descriptorsA = constructDescriptors(samplesA);
  ShapeDescriptors descriptorsB = constructDescriptors(samplesB);
  cv::Mat cost_matrix = computeCostMatrix(descriptorsA, descriptorsB,
                                          epsilonCost, write_images, filePath, filePostFix);

  // save the result
  if (write_images)
  {
    std::stringstream image_a_path_raw;
    image_a_path_raw << filePath << "/edge_imageA_raw" << filePostFix << ".bmp";
    std::stringstream image_b_path_raw;
    image_b_path_raw << filePath << "/edge_imageB_raw" << filePostFix << ".bmp";
    cv::imwrite(image_a_path_raw.str().c_str(), edge_imageA_raw);
    cv::imwrite(image_b_path_raw.str().c_str(), edge_imageB_raw);
    std::stringstream image_a_path;
    image_a_path << filePath << "/edge_imageA" << filePostFix << ".bmp";
    std::stringstream image_b_path;
    image_b_path << filePath << "/edge_imageB" << filePostFix << ".bmp";
    cv::imwrite(image_a_path.str().c_str(), edge_imageA);
    cv::imwrite(image_b_path.str().c_str(), edge_imageB);
  }

  // do bipartite graph matching to find point correspondences
  // (uses code from http://www.magiclogic.com/assignment.html)
  Path min_path;
  double score = getMinimumCostPath(cost_matrix, min_path);
  displayMatch(edge_imageA, edge_imageB, samplesA, samplesB, min_path, max_displacement,
               filePath, filePostFix);
  int sizeA = samplesA.size();
  int sizeB = samplesB.size();

  // TODO: Return correspondences as well
  return (score-(fabs(sizeA-sizeB)*epsilonCost));
}

ShapeDescriptors extractDescriptors(cv::Mat& image)
{
  cv::Mat edge_image(image.size(), image.type());
  cv::Mat edge_image_raw;

  // do edge detection
  cv::Canny(image, edge_image, 0.05, 0.5);
  edge_image.copyTo(edge_image_raw);

  // sample a subset of the edge pixels
  Samples samples = samplePoints(edge_image);

  // construct shape descriptors for each sample
  ShapeDescriptors descriptors = constructDescriptors(samples);
  return descriptors;
}

Samples samplePoints(cv::Mat& edge_image, double percentage)
{
  Samples samples;
  Samples all_points;
  cv::Scalar pixel;
  for (int y=0; y < edge_image.rows; y++)
  {
    for (int x=0; x < edge_image.cols; x++)
    {
      if (edge_image.at<uchar>(y, x) > 0)
      {
        all_points.push_back(cv::Point(x, y));
      }
    }
  }

  // set edge image to black
  edge_image = cv::Scalar(0);

  // subsample a percentage of all points
  int scale = 1 / percentage;
  for (unsigned int i=0; i < all_points.size(); i++)
  {
    if (i%scale == 0)
    {
      samples.push_back(all_points.at(i));
      edge_image.at<uchar>(samples.back().y, samples.back().x) = 255;
    }
  }
  return samples;
}

int getHistogramIndex(double radius, double theta, int radius_bins, int theta_bins)
{
  int radius_idx = std::max(std::min((int)(radius*radius_bins), radius_bins), 0);
  int theta_idx = std::max(std::min((int)(theta*theta_bins), theta_bins),0);
  int idx = theta_idx*radius_bins+radius_idx;
  return idx;
}

ShapeDescriptors constructDescriptors(Samples2f& samples,
                                      cv::Point2f& center,
                                      int radius_bins,
                                      int theta_bins,
                                      double max_radius)
{
  ShapeDescriptors descriptors;
  ShapeDescriptor descriptor;
  // double max_radius = 0;
  double radius, theta;
  double x1, x2, y1, y2;
  unsigned int i, j, k, m;

  // find maximum radius for normalization purposes (unless passed in as argument)
  if (max_radius == 0)
  {
    for (i=0; i < samples.size(); i++)
    {
      x1 = samples.at(i).x;
      y1 = samples.at(i).y;
      for (k=0; k < samples.size(); k++)
      {
        if (k != i)
        {
          x2 = samples.at(k).x;
          y2 = samples.at(k).y;

          radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
          if (radius > max_radius)
          {
            max_radius = radius;
          }
        }
      }
    }
  }
  max_radius = log(max_radius);

  // build a descriptor for each sample
  for (i=0; i < samples.size(); i++)
  {
    // initialize descriptor
    descriptor.clear();
    for (j=0; j < radius_bins*theta_bins; j++)
    {
      descriptor.push_back(0);
    }

    // Orient descriptor towards center
    double center_angle = atan2(center.y-samples[i].y, center.x-samples[i].x);
    x1 = samples.at(i).x;
    y1 = samples.at(i).y;

    // construct descriptor
    for (m=0; m < samples.size(); m++)
    {
      if (m != i)
      {
        x2 = samples.at(m).x;
        y2 = samples.at(m).y;
        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        radius = log(radius);
        radius /= max_radius;
        theta = atan2(y1-y2,x1-x2) + M_PI/2;
        // TODO: Rotate theta so that center orientation is 0
        // Get theta in range [0,1]
        theta /= 2*M_PI;
        // FIXME: It's broken here
        int idx = getHistogramIndex(radius, theta, radius_bins, theta_bins);
        descriptor.at(idx)++;
      }
    }

    // add descriptor to std::vector of descriptors
    descriptors.push_back(descriptor);
  }

  return descriptors;
}

ShapeDescriptors constructDescriptors(Samples& samples,
                                      unsigned int radius_bins,
                                      unsigned int theta_bins)
{
  ShapeDescriptors descriptors;
  ShapeDescriptor descriptor;
  double max_radius = 0;
  double radius, theta;
  double x1, x2, y1, y2;
  unsigned int i, j, k, m;

  // find maximum radius for normalization purposes
  for (i=0; i < samples.size(); i++)
  {
    for (k=0; k < samples.size(); k++)
    {
      if (k != i)
      {
        x1 = samples.at(i).x;
        y1 = samples.at(i).y;
        x2 = samples.at(k).x;
        y2 = samples.at(k).y;

        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        if (radius > max_radius)
        {
          max_radius = radius;
        }
      }
    }
  }
  max_radius = log(max_radius);

  // build a descriptor for each sample
  for (i=0; i < samples.size(); i++)
  {
    // initialize descriptor
    descriptor.clear();
    for (j=0; j < radius_bins*theta_bins; j++)
    {
      descriptor.push_back(0);
    }

    // construct descriptor
    for (m=0; m < samples.size(); m++)
    {
      if (m != i)
      {
        x1 = samples.at(i).x;
        y1 = samples.at(i).y;
        x2 = samples.at(m).x;
        y2 = samples.at(m).y;

        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        radius = log(radius);
        radius /= max_radius;
        theta = atan(fabs(y1-y2) / fabs(x1-x2));
        theta += M_PI/2;
        if (y1-y2 < 0)
        {
          theta += M_PI;
        }
        theta /= 2*M_PI;
        int idx = getHistogramIndex(radius, theta, radius_bins, theta_bins);
        descriptor.at(idx)++;
      }
    }

    // add descriptor to std::vector of descriptors
    descriptors.push_back(descriptor);
  }

  return descriptors;
}

cv::Mat computeCostMatrix(ShapeDescriptors& descriptorsA,
                          ShapeDescriptors& descriptorsB,
                          double epsilonCost,
                          bool write_images,
                          std::string filePath, std::string filePostFix)
{
  int mat_size = std::max(descriptorsA.size(), descriptorsB.size());
  cv::Mat cost_matrix(mat_size, mat_size, CV_64FC1, 0.0f);
  double d_cost, hi, hj;
  ShapeDescriptor& descriptorA = descriptorsA.front();
  ShapeDescriptor& descriptorB = descriptorsB.front();

  // initialize cost matrix for dummy values
  for (int i=0; i < cost_matrix.rows; i++)
  {
    for (int j=0; j < cost_matrix.cols; j++)
    {
      cost_matrix.at<double>(i,j) = epsilonCost;
    }
  }

  for (unsigned int i=0; i < descriptorsA.size(); i++)
  {
    descriptorA = descriptorsA.at(i);
    for (unsigned int j=0; j < descriptorsB.size(); j++)
    {
      descriptorB = descriptorsB.at(j);
      d_cost = 0;

      // compute cost between shape context i and j
      // using chi-squared test statistic
      for (unsigned int k=0; k < descriptorA.size(); k++)
      {
        hi = descriptorA.at(k) / (descriptorsA.size() - 1); // normalized bin val
        hj = descriptorB.at(k) / (descriptorsB.size() - 1); // normalized bin val
        if (hi + hj > 0)
        {
          d_cost += pow(hi-hj, 2) / (hi + hj);
        }
      }
      d_cost /= 2;
      cost_matrix.at<double>(i,j) = d_cost;
    }
  }

  cv::Mat int_cost_matrix;
  cost_matrix.convertTo(int_cost_matrix, CV_8UC1, 255);
  if (write_images)
  {
    std::stringstream cost_matrix_path;
    cost_matrix_path << filePath << "/cost_matrix" << filePostFix << ".bmp";
    cv::imwrite(cost_matrix_path.str().c_str(), int_cost_matrix);
  }

  return cost_matrix;
  // return int_cost_matrix;
}

double getMinimumCostPath(cv::Mat& cost_matrix, Path& path)
{
  const int dim = cost_matrix.rows;
  LapCost **cost_mat;
  cost_mat = new LapCost*[dim];
  // std::cout << "Allocating cost matrix" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    cost_mat[r] = new LapCost[dim];
  }
  // std::cout << "Populating cost matrix" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    for (int c = 0; c < dim; ++c)
    {
      cost_mat[r][c] = cost_matrix.at<double>(r,c);
    }
  }
  LapRow* rowsol;
  LapCol* colsol;
  LapCost* u;
  LapCost* v;
  rowsol = new LapCol[dim];
  colsol = new LapRow[dim];
  u = new LapCost[dim];
  v = new LapCost[dim];
  // std::cout << "Running lap" << std::endl;
  LapCost match_cost = lap(dim, cost_mat, rowsol, colsol, u, v);
  // std::cout << "Ran lap" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    int c = rowsol[r];
    path.push_back(c);
  }
  // std::cout << "Converted lap result" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    delete cost_mat[r];
  }
  delete cost_mat;
  delete u;
  delete v;
  delete rowsol;
  delete colsol;
  return match_cost;
}

void displayMatch(cv::Mat& edge_imageA, cv::Mat& edge_imageB,
                  Samples& samplesA, Samples& samplesB,
                  Path& path, int max_displacement, std::string filePath,
                  std::string filePostFix)
{
  int a_row_offset = 0;
  int b_row_offset = 0;
  int a_col_offset = 0;
  int b_col_offset = 0;
  if (edge_imageA.rows > edge_imageB.rows)
  {
    b_row_offset = (edge_imageA.rows - edge_imageB.rows)/2;
  }
  else if (edge_imageA.rows < edge_imageB.rows)
  {
    a_row_offset = (edge_imageB.rows - edge_imageA.rows)/2;
  }
  if (edge_imageA.cols > edge_imageB.cols)
  {
    b_col_offset = (edge_imageA.cols - edge_imageB.cols)/2;
  }
  else if (edge_imageA.cols < edge_imageB.cols)
  {
    a_col_offset = (edge_imageB.cols - edge_imageA.cols)/2;
  }
  cv::Mat disp_img(std::max(edge_imageA.rows, edge_imageB.rows),
                   std::max(edge_imageA.cols, edge_imageB.cols), CV_8UC3, cv::Scalar(0,0,0));
  // Display image A
  for (int r=0; r < edge_imageA.rows; r++)
  {
    for (int c=0; c < edge_imageA.cols; c++)
    {
      if (edge_imageA.at<uchar>(r,c) > 0)
        disp_img.at<cv::Vec3b>(r+a_row_offset, c+a_col_offset) = cv::Vec3b(255,0,0);
    }
  }

  // Display image B as another color
  for (int r=0; r < edge_imageB.rows; r++)
  {
    for (int c=0; c < edge_imageB.cols; c++)
    {
      if (edge_imageB.at<uchar>(r,c) > 0)
        disp_img.at<cv::Vec3b>(r+b_row_offset, c+b_col_offset) = cv::Vec3b(0,0,255);
    }
  }

  for (unsigned int i = 0; i < samplesA.size(); ++i)
  {
    cv::Point start_point = samplesA[i];
    start_point.x += a_col_offset;
    start_point.y += a_row_offset;
    cv::Point end_point = samplesB[path[i]];
    end_point.x += b_col_offset;
    end_point.y += b_row_offset;
    if (std::abs(start_point.x - end_point.x) +
        std::abs(start_point.y - end_point.y) < max_displacement &&
        end_point.x > 0 && end_point.x < disp_img.rows &&
        end_point.y > 0 && end_point.x < disp_img.cols)
    {
      cv::line(disp_img, start_point, end_point, cv::Scalar(0,255,0));
    }
  }
  std::stringstream match_path;
  match_path << filePath << "/matches" << filePostFix << ".bmp";
  cv::imwrite(match_path.str().c_str(), disp_img);
  cv::imshow("match", disp_img);
  cv::imshow("edgesA", edge_imageA);
  cv::imshow("edgesB", edge_imageB);
  cv::waitKey(3);
}
};
