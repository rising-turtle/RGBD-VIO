/*
	Dec. 10, 2020, He Zhang, hzhang8@vcu.edu

	use plane to correct pose  

*/

#pragma once 

#include <list>
#include <algorithm>
#include <map>
#include <vector>
#include <numeric>
#include <set>
#include "tf/tf.h"
#include <eigen3/Eigen/Dense>

class RVIO; 

class PlaneCorrectPose
{
public: 
	PlaneCorrectPose(); 
	~PlaneCorrectPose();

	void reset(); 
	bool correctPose(const RVIO& ); 

	tf::Transform getCurrPose(){
		return currCorrPose; 
	}

private:

	tf::Transform prevPose; 
	tf::Transform currPose; 
	tf::Transform prevCorrPose; 
	tf::Transform currCorrPose; 
	double prevTime; 
	double currTime; 	
	bool isFloorDetected; 
	Eigen::Matrix<double, 4, 1> floorPlane; 

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};
