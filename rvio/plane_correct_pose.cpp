/*
	Dec. 10, 2020, He Zhang, hzhang8@vcu.edu

	use plane to correct pose  

*/


#include "plane_correct_pose.h"
// #include "parameters.h"

#include "rvio.h"

using namespace Eigen;
using namespace std;

PlaneCorrectPose::PlaneCorrectPose()
{
	reset(); 
}

PlaneCorrectPose::~PlaneCorrectPose()
{

}

void PlaneCorrectPose::reset()
{
	prevTime = -1; 
	currTime = -1; 
	isFloorDetected = false; 
}

bool PlaneCorrectPose::correctPose(const RVIO& estimator)
{
	if(!isFloorDetected && estimator.mbFirstFloorObserved){
		isFloorDetected = true; 
		floorPlane = estimator.fp_Pls; 
	}

	if(prevTime == -1){
		prevTime = currTime = estimator.Headers[estimator.frame_count]; 
		prevCorrPose = currCorrPose = prevPose = currPose = estimator.mCurrIMUPose; 
		return false; 
	}
	// else 

	// 1. predict current pose 
	currTime = estimator.Headers[estimator.frame_count]; 
	currPose = estimator.mCurrIMUPose; 

	tf::Transform inc_pose = prevPose.inverse() * currPose; 
	tf::Transform currCorrPose = prevCorrPose * inc_pose; 

	if(estimator.bPls[estimator.frame_count] == false){ // no way to update it. 
		// ROS_ERROR("what frame_count: %i and no floor plane is founded ", estimator.frame_count); 
		return false; 
	}
	
	Eigen::Matrix<double, 4, 1> local_plane = estimator.Pls[estimator.frame_count]; 

	// 2. compute distance 
	Eigen::Matrix<double, 3, 1> nv_g = floorPlane.block<3,1>(0,0) ; // (floorPlane(0), floorPlane(1), floorPlane(2)); 
	Eigen::Matrix<double, 3, 1> Pi(currCorrPose.getOrigin().getX(), 
									currCorrPose.getOrigin().getY(),
									currCorrPose.getOrigin().getZ()); 
	double dg = floorPlane(3); 

	double dl_pred = nv_g.dot(Pi) + dg; 
	double dl_meas = local_plane(3); 
	double residual = dl_meas - dl_pred; 

	ROS_ERROR("plane_correct_pose.cpp: before correct, residual : %lf ", residual); 

	// 
	Eigen::Vector3d delta_pi = residual * nv_g; 
	dl_pred = nv_g.dot(Pi) + nv_g.dot(delta_pi) + dg; 
	residual = dl_meas - dl_pred; 

	tf::Vector3 new_pi(Pi(0) + delta_pi(0), Pi(1) + delta_pi(1), Pi(2) + delta_pi(2)); 
	currCorrPose.setOrigin(new_pi); 

	ROS_DEBUG("Pose_i: %f %f %f, corrected_Pose_i: %f %f %f", Pi(0), Pi(1), Pi(2), 
			new_pi.getX(), new_pi.getY(), new_pi.getZ());

	ROS_WARN("plane_correct_pose.cpp: after correct, residual : %lf ", residual); 

	// update the pose 

	// for next loop 
	prevTime = currTime; 
	prevPose = currPose; 
	prevCorrPose = currCorrPose; 

	return true;
}