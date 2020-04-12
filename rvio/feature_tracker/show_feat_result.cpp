/*
	Apr. 11, 2020, He Zhang, hzhang8@vcu.edu 

	extract features and show them in point cloud 

*/

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <vector>
#include <string>
#include <iostream>
#include "vtk_viewer.h"

using namespace std; 

void generatePointCloud(cv::Mat& rgb, cv::Mat& dpt, pcl::PointCloud<pcl::PointXYZRGB>& pc); 

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "show_feat_result");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);  // Debug Info  

    if(argc < 3){
    	ROS_INFO("usage: ./show_feat_result [rgb] [dpt]"); 
    	return -1;
    }

    // read rgb and dpt image 
    cv::Mat bgr = cv::imread(argv[1], -1); 
    cv::Mat dpt = cv::imread(argv[2], -1); 

    if(bgr.empty()){
    	ROS_ERROR("show_feat_result: failed to load rgb file %s", argv[1]); 
    	return -1; 
    }
    if(dpt.empty()){
    	ROS_ERROR("show_feat_result: failed to load dpt file %s", argv[2]); 
    	return -1; 
    }

    // extract features and show it 
    cv::Mat rgb, gray, mask; 
    int MIN_DIST = 10; 
    int MAX_CNT = 300; 
    vector<cv::Point2f> n_pts;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY); 
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB); 
	cv::goodFeaturesToTrack(gray, n_pts, MAX_CNT, 0.01, MIN_DIST, mask);
	
	// find depth valid point 
	std::vector<bool> vv(n_pts.size());
	double MAX_DPT_RANGE = 7.0; 
	for(int i=0; i<vv.size(); i++){
		float ui = n_pts[i].x; // ui 
        float vi = n_pts[i].y; // vi

        float d = (float)dpt.at<unsigned short>(std::round(vi), std::round(ui)) * 0.001;
        if(d>= 0.6 && d <= MAX_DPT_RANGE) {
            vv[i] = true; 
        }else{
            vv[i] = false; // make it an invalid depth value 
        }
	}

	// highlight the extracted feature points 
	for(int i=0; i<n_pts.size(); i++){

		if(vv[i]) // valid 
			cv::circle(rgb, n_pts[i], 2, cv::Scalar(0, 0, 255), 2); 
		else
			cv::circle(rgb, n_pts[i], 2, cv::Scalar(255, 0, 0), 2); 
	}
	cv::imshow("show it", rgb);
	cv::imwrite("feature_result.png", rgb);
	// cv::waitKey(0);	

	// generate point cloud 
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>); 
	generatePointCloud(bgr, dpt, *pc); 

	// save pci 
    pcl::io::savePCDFile(string("point_cloud.pcd"), *pc); 

    // markColor(*pci, pni->mv_indices[best_i], GREEN); 
    // markColor(*pcj_ni, pnj->mv_indices[best_j], RED); 
    // markColor(*pci,  GREEN); 
    // markColor(*pcj_ni, RED); 
 
    CVTKViewer<pcl::PointXYZRGB> v;
    // v.getViewer()->addCoordinateSystem(0.2, 0, 0); 
    v.addPointCloud(pc, "pci"); 

    // add feature circles to the point cloud 
	double f = 459.5; 
  	double cx = 332.7; 
  	double cy = 259.0; 
	for(int i=0; i<n_pts.size(); i++){
		if(vv[i]){
			float ui = n_pts[i].x; // ui 
        	float vi = n_pts[i].y; // vi
        	float d = (float)dpt.at<unsigned short>(std::round(vi), std::round(ui)) * 0.001;
			pcl::PointXYZRGB pt; 
			pt.z = d; 
			pt.x = (ui - cx)/f * d; 
			pt.y = (vi - cy)/f * d; 
			string name{i+1+'0'}; 
			v.getViewer()->addSphere(pt, 0.05, 1.0, 0., 0., name); 
		}
	}

    while(ros::ok() && !v.stopped())
    {
      v.runOnce(); 
      usleep(100*1000); 
    } 

	return 0; 
}


void generatePointCloud(cv::Mat& rgb, cv::Mat& dpt, pcl::PointCloud<pcl::PointXYZRGB>& pc) 
{  
  double z; 
  double px, py, pz; 
  int skip = 1; 
  int height = rgb.rows/skip; 
  int width = rgb.cols/skip; 
  int N = (rgb.rows/skip)*(rgb.cols/skip); 
  double dpt_scale = 0.001; 

  pc.points.reserve(N); 
  // pc.width = width; 
  // pc.height = height; 

  unsigned char r, g, b; 
  int pixel_data_size = 3; 
  if(rgb.type() == CV_8UC1)
  {
    pixel_data_size = 1; 
  }
  
  int color_idx; 
  char red_idx = 2, green_idx =1, blue_idx = 0;

  double f = 459.5; 
  double cx = 332.7; 
  double cy = 259.0; 
  // Point pt; 
  pcl::PointXYZRGB pt;
  for(int v = 0; v<rgb.rows; v+=skip)
  for(int u = 0; u<rgb.cols; u+=skip)
  {
    // Point& pt = pc.points[v*width + u]; 
    z = dpt.at<unsigned short>((v), (u))*dpt_scale;
    if(std::isnan(z) || z <= 0) 
    {
      pt.x = std::numeric_limits<float>::quiet_NaN();  
      pt.y = std::numeric_limits<float>::quiet_NaN();  
      pt.z = std::numeric_limits<float>::quiet_NaN();  
      continue; 
    }

 	px = (u - cx)/f * z;
 	py = (v - cy)/f * z; 
 	pz = z;  

    pt.x = px;  pt.y = py;  pt.z = pz; 
    color_idx = (v*rgb.cols + u)*pixel_data_size;
    if(pixel_data_size == 3)
    {
      r = rgb.at<uint8_t>(color_idx + red_idx);
      g = rgb.at<uint8_t>(color_idx + green_idx); 
      b = rgb.at<uint8_t>(color_idx + blue_idx);
    }else{
      r = g = b = rgb.at<uint8_t>(color_idx); 
    }
    pt.r = r; pt.g = g; pt.b = b; 
    pc.points.push_back(pt); 
  }
  pc.height = 1; 
  pc.width = pc.points.size(); 
  return ;

}