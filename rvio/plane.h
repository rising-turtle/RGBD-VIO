/*
    Aug. 25 He Zhang, hzhang8@vcu.edu 

    extract plane from point cloud and 
    
    plane related functions 
*/

#pragma once
#include <Eigen/Core>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <opencv2/core/eigen.hpp>

class Plane
{
public:
    Plane(); 
    virtual ~Plane(); 
    
    // plane parameters 
    // Eigen::Vector3d m_nv; 
    // double m_d; 
    
    template<typename PointT>
    void computeByPCL(boost::shared_ptr<pcl::PointCloud<PointT> >& in, 
	pcl::PointIndices::Ptr& inliers, Eigen::Vector3d& nv, double& d, double dis_threshold = 0.01); 

};

#include "plane.hpp"
