
/*
    Aug. 25 He Zhang, hzhang8@vcu.edu 

    extract plane from point cloud and 
    
    plane related functions 
*/

template<typename PointT>
void Plane::computeByPCL(boost::shared_ptr<pcl::PointCloud<PointT> >& in, pcl::PointIndices::Ptr& inliers, Eigen::Vector3d& nv, double& d, double dis_threshold)
{
    pcl::SACSegmentation<PointT> seg; 
    seg.setOptimizeCoefficients(true); 
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC); 
    seg.setDistanceThreshold(dis_threshold); 
    
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); 
    seg.setInputCloud(in); 
    seg.segment(*inliers, *coefficients); 
    
    nv(0) = coefficients->values[0]; 
    nv(1) = coefficients->values[1];  
    nv(2) = coefficients->values[2]; 
    d = coefficients->values[3]; 
    
    if(nv(2) < 0)
    {
	   nv *= -1.;
       d *= -1.;
    }
    return ; 
}
