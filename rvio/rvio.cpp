/*
    Aug. 21 2018, He Zhang, hzhang8@vcu.edu 
    
    RVIO: tightly couple features [no depth, triangulated depth from stereo or sfm] 
    with IMU integration 

*/

#include "rvio.h"
// #include "../vo/stereo_triangulate.h"
#include "../utility/utility.h"
#include "opencv/cv.h"
#include <iostream>
#include <string>
#include <thread>
#include "../utility/tic_toc.h"
#include "projection_quat.h"
#include "visualization.h"
#include "depth_factor.h"
#include "initial/gmm_model.h"
// #include "plane.h"

using namespace QUATERNION_VIO; 
using namespace Eigen;


namespace{

    void printTF(tf::Transform& T, string name="")
    {
	tf::Quaternion q = T.getRotation(); 
	tf::Vector3 t = T.getOrigin(); 
	cout <<name<<" "<<t.getX()<<" "<<t.getY()<<" "<<t.getZ()<<" "<<q.getX()<<" "<<q.getY()<<" "<<q.getZ()<<" "<<q.getW()<<endl; 
    }
}

RVIO::RVIO():
mbFirstIMU(true),
mbInited(false),
frame_count(0),
mbStereo(false),
f_manager(Rs),
m_imgCnt(0),
mp_gmm(new GMM_Model),
mPCNoFloor(new pcl::PointCloud<pcl::PointXYZI>),
mPCFloor(new pcl::PointCloud<pcl::PointXYZ>),
mDepthPC(new pcl::PointCloud<pcl::PointXYZI>),
mZoomDis(10.),
mKDTree(new pcl::KdTreeFLANN<pcl::PointXYZI>),
mFloorZ(NOT_INITIED),
mFloorRange(0.15),
mbFirstFloorObserved(false)
{
    clearState();
}
RVIO::~RVIO(){

    // wait for other threads? 
    if(mp_gmm) delete mp_gmm;
}

void RVIO::clearState()
{
    m_process.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    currTime = 0;

    mbInitFirstPoseFlag = false;

    for(int i=0; i<WN + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();

        linear_acceleration_buf[i].clear(); 
        angular_velocity_buf[i].clear(); 
        if(pre_integrations[i] != NULL)
            delete pre_integrations[i]; 
        pre_integrations[i] = NULL; 

        // plane related parameters 
        bPls[i] = false; 
        Pls[i].setZero(); 
    }

    // initially the global plane's parameter is set as [0, 0, 1]
    fp_Pls.setZero(); fp_Pls[2] = 1;  

    tic[0] = Vector3d::Zero(); 
    ric[0] = Matrix3d::Identity(); 

    frame_count = 0;
    solver_flag = INITIAL;
    all_image_frame.clear(); 
    
    R_imu = Eigen::Matrix3d::Identity(); 

    if(tmp_pre_integration != NULL)
	    delete tmp_pre_integration; 
    tmp_pre_integration = NULL; 

    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();

    // failure_occur = 0;
    initial_timestamp = 0;
    m_process.unlock(); 
}

void RVIO::associateDepthGMM(map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>& featureFrame, const cv::Mat& dpt)
{
    // x, y, z, p_u, p_v, velocity_x, velocity_y;
    map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>::iterator it = featureFrame.begin(); 
    while(it != featureFrame.end()){

        float nor_ui = it->second[0].second(0); 
        float nor_vi = it->second[0].second(1); 
        float ui = it->second[0].second(3); // ui 
        float vi = it->second[0].second(4); // vi

        double d = (double)dpt.at<unsigned short>(std::round(vi), std::round(ui)) * 0.001;
        int use_sim = 0; 
        if(d>= 0.3 && d <= MAX_DPT_RANGE) {
            // it->second[0].second(2) = d;

            double mu_d, mu_l, sig_d, sig_l; 
            mp_gmm->gmm_model_depth(std::round(vi), std::round(ui), dpt, mu_d, sig_d, use_sim); 
            mp_gmm->gmm_model_inv_depth(std::round(vi), std::round(ui), dpt, mu_l, sig_l, use_sim); 

            it->second[0].second(2) = mu_d; 
            it->second[0].second(7) = mu_l; 
            it->second[0].second(8) = sig_d; 
            it->second[0].second(9) = sig_l; 

        }else{
            it->second[0].second(2) = 0.; // make it an invalid depth value 
        }
        it++;
    }
    return ; 
}

void RVIO::associateDepthSimple(map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>& featureFrame, const cv::Mat& dpt)
{
    // x, y, z, p_u, p_v, velocity_x, velocity_y;
    map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>::iterator it = featureFrame.begin(); 
    while(it != featureFrame.end()){

        float nor_ui = it->second[0].second(0); 
        float nor_vi = it->second[0].second(1); 
        float ui = it->second[0].second(3); // ui 
        float vi = it->second[0].second(4); // vi

        float d = (float)dpt.at<unsigned short>(std::round(vi), std::round(ui)) * 0.001;
        if(d>= 0.3 && d <= MAX_DPT_RANGE) {
            it->second[0].second(2) = d;
        }else{
            it->second[0].second(2) = 0.; // make it an invalid depth value 
        }
        it++;
    }
    return ; 
}

void RVIO::setInputDepthImage(const cv::Mat& dpt_img)
{   
    // use input depth image to compute the current point cloud and kd tree 
    // median filter to reduce noise 
    cv::Mat dst; 
    cv::medianBlur(dpt_img, dst, 3);  

    // clear current point cloud 
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI> > tmpPC(new pcl::PointCloud<pcl::PointXYZI>()); 
    // mPC->clear(); 
    mDepthPC.swap(tmpPC);

    // 2. compute point cloud 
    pcl::PointXYZI pt;

    // need FX, FY, CX, CY for depth image 
    for(int r = 0; r < dst.rows; r++)
    for(int c = 0; c < dst.cols; c++){
        float d = (float)dst.at<unsigned short>(r, c) * 0.001;
        if(d >= 0.3 && d <= 5.0){//MAX_DPT_RANGE){
            // valid depth value 
            pt.x = (c - DCX)/DFX * mZoomDis; 
            pt.y = (r - DCY)/DFY * mZoomDis; 
            pt.z = mZoomDis; 
            pt.intensity = d; 
            mDepthPC->points.push_back(pt);
        }
    }
    mDepthPC->height = 1;
    mDepthPC->width = mDepthPC->points.size(); 

    // 2. set kd search tree
    mKDTree->setInputCloud(mDepthPC); 
    return ; 
}

void RVIO::associateDepthInterporlate(map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>& featureFrame, const cv::Mat& dpt)
{
    // build kd search tree for the given input depth 
    setInputDepthImage(dpt);     

    // find out depth of the visual features 
    pcl::PointXYZI query_pt; 

    // x, y, z, p_u, p_v, velocity_x, velocity_y;
    map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>::iterator it = featureFrame.begin(); 
    while(it != featureFrame.end()){

        float nor_ui = it->second[0].second(0); 
        float nor_vi = it->second[0].second(1); 
        float ui = it->second[0].second(3); // ui 
        float vi = it->second[0].second(4); // vi

        query_pt.x = mZoomDis * nor_ui; 
        query_pt.y = mZoomDis * nor_vi; 
        query_pt.z = mZoomDis; 
        double query_pt_depth; 

        if(mDepthPC->width >= 10){
            // kd search 
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqrDis;

            mKDTree->nearestKSearch(query_pt, 3, pointSearchInd, pointSearchSqrDis); 
            double minDepth, maxDepth; 
            if(pointSearchSqrDis[0] < 0.5 && pointSearchInd.size() == 3){

                pcl::PointXYZI depthPoint = mDepthPC->points[pointSearchInd[0]];
                double x1 = depthPoint.x * depthPoint.intensity / mZoomDis;
                double y1 = depthPoint.y * depthPoint.intensity / mZoomDis;
                double z1 = depthPoint.intensity;
                minDepth = z1;
                maxDepth = z1;

                depthPoint = mDepthPC->points[pointSearchInd[1]];
                double x2 = depthPoint.x * depthPoint.intensity / mZoomDis;
                double y2 = depthPoint.y * depthPoint.intensity / mZoomDis;
                double z2 = depthPoint.intensity;
                minDepth = (z2 < minDepth)? z2 : minDepth;
                maxDepth = (z2 > maxDepth)? z2 : maxDepth;

                depthPoint = mDepthPC->points[pointSearchInd[2]];
                double x3 = depthPoint.x * depthPoint.intensity / mZoomDis;
                double y3 = depthPoint.y * depthPoint.intensity / mZoomDis;
                double z3 = depthPoint.intensity;
                minDepth = (z3 < minDepth)? z3 : minDepth;
                maxDepth = (z3 > maxDepth)? z3 : maxDepth;

                double u = nor_ui;
                double v = nor_vi;

                // intersection point between direction of OP and the Plane formed by [P1, P2, P3]
                // query_pt_depth = (x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1) 
                // / (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2 + u*y1*z2 - u*y2*z1
                //     - v*x1*z2 + v*x2*z1 - u*y1*z3 + u*y3*z1 + v*x1*z3 - v*x3*z1 + u*y2*z3 
                //     - u*y3*z2 - v*x2*z3 + v*x3*z2);

                double up = (x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1) ; 
                double down = (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2 + u*y1*z2 - u*y2*z1
                    - v*x1*z2 + v*x2*z1 - u*y1*z3 + u*y3*z1 + v*x1*z3 - v*x3*z1 + u*y2*z3 
                    - u*y3*z2 - v*x2*z3 + v*x3*z2); 

                if(down <= 1e-5){
                    query_pt_depth = 0;
                    if(z1 == z2 == z3)
                        query_pt_depth = z1; 
                }
                else
                    query_pt_depth = up/down; 

                // check the validity of the depth measurement 
                if(maxDepth - minDepth > 2) // lie on an edge or noisy point? 
                {   
                    query_pt_depth = 0; 
                }else if(query_pt_depth - maxDepth > 0.2)
                {
                    query_pt_depth = maxDepth; 
                }else if(query_pt_depth - minDepth < - 0.2)
                {
                    query_pt_depth = minDepth; 
                }

                // if((query_pt_depth < 0.3 || query_pt_depth > 5)){
                    // ROS_WARN("what? query_pt_depth: %lf minDepth: %lf maxDepth: %lf", query_pt_depth, minDepth, maxDepth);
                // }

                if(query_pt_depth != query_pt_depth){
                    ROS_ERROR("what? query_pt_depth is NaN, x1: %f y1: %f z1: %f x2: %f y2: %f z2: %f x3: %f y3: %f z3: %f u: %f v:%f",
                             x1, y1, z1, x2, y2, z2, x3, y3, z3, u, v);
                    ROS_WARN("up: %lf down: %lf", up, down);
                }

            }else{
                query_pt_depth = 0; 
            }
        }else{
            query_pt_depth = 0; 
        }

      

        it->second[0].second(2) = query_pt_depth; 
        it++;
    }

    return ; 
}


void RVIO::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    mbInitFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}


void RVIO::processIMU(double dt, Vector3d & linear_acceleration, Vector3d& angular_velocity)
{
    if(mbFirstIMU)
    {
    	mbFirstIMU = false; 
    	acc_0 = linear_acceleration; 
    	gyr_0 = angular_velocity;
    }

    if(!pre_integrations[frame_count])
    {
    	pre_integrations[frame_count] = new IntegrationBase(acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]); 
    }
    if(frame_count != 0)
    {
        // cout<<"RVIO.cpp: processIMU frame_count: "<<frame_count<<" dt: "<<dt<<" linear_acceleration: "<<linear_acceleration.transpose()<<" angular_velocity: "<<angular_velocity.transpose()<<endl;
    	pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); 

    	tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); 
    	dt_buf[frame_count].push_back(dt); 
    	linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    	angular_velocity_buf[frame_count].push_back(angular_velocity);

    	int j = frame_count; 
    	Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - mg; 
    	Vector3d un_gyr = 0.5 *(gyr_0 + angular_velocity) - Bgs[j]; 
    	Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); 
    	// R_imu *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); 
    	Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - mg;
    	Vector3d un_acc = 0.5 *(un_acc_0 + un_acc_1); 
    	Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; 
    	Vs[j] += dt * un_acc; 
        // cout<<" dt: "<<dt<<" un_acc: "<<un_acc.transpose()<<" Vs["<<j<<"]: "<<Vs[j].transpose()<<" g: "<<mg.transpose()<<endl;
    }else // save for initialization 
    {
    	// linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    	// angular_velocity_buf[frame_count].push_back(angular_velocity); 
    }
    acc_0 = linear_acceleration; 
    gyr_0 = angular_velocity; 
}


void RVIO::setParameter()
{
    m_process.lock(); 
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
	   tic[i] = TIC[i];
	   ric[i] = RIC[i];
    }
    Eigen::Quaterniond q(ric[0]); 
    mTIC = tf::Transform(tf::Quaternion(q.x(), q.y(), q.z(), q.w()), tf::Vector3(tic[0][0], tic[0][1], tic[0][2])); 
    printTF(mTIC, "RVIO.cpp: initial mTIC: ");
    f_manager.setRic(ric);
    mg = G; 

    m_process.unlock(); 
}

// void RVIO::processImage_Init(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
void RVIO::processImage_Init(const map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>> &image, const double header)
{
    ROS_DEBUG("timestamp %lf with feature points %lu", header, image.size());

    // if(f_manager.addFeatureCheckParallax(frame_count, image))
    if(f_manager.addFeatureCheckParallaxSigma(frame_count, image))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_INFO("handle frame at timstamp %lf is a %s", header, marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_INFO("timestamp %lf number of feature: %d", header, f_manager.getFeatureCount());
    Headers[frame_count] = header; 

    // copy image to tmp_image 
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> tmp_image; 
    map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>>::const_iterator it = image.begin(); 
    while(it != image.end()){
        vector<pair<int, Eigen::Matrix<double, 7, 1>>> tV; 
        for(int i=0; i<it->second.size(); i++){
            Eigen::Matrix<double, 7, 1> tM; 

            for(int j=0; j<7; j++){
                tM(j) = it->second[i].second(j); 
            }
            tV.push_back(make_pair(it->second[i].first, tM)); 
        }
        //tmp_image[it->first] = tV; 
        tmp_image.emplace(it->first, tV); 
        it++; 
    }


    ImageFrame imageframe(tmp_image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(solver_flag == INITIAL){
        // cout<<"RVIO.cpp: at frame_count: "<<frame_count<<" feature_manager has: "<<f_manager.feature.size()<<" features!"<<endl; 
        if(frame_count == WN){
            bool result = false; 
            if((header - initial_timestamp) > 0.1)
            {
                result = initialStructure(); 
                initial_timestamp = header; 
            }
            if(result) // succeed to initialize 
            {
                solver_flag = NON_LINEAR; 

                // now only debug initialization 
                ROS_INFO("Initialization finish!");
                showStatus();

                // f_manager.triangulateSimple(frame_count, Ps, Rs, tic, ric);
                f_manager.triangulateWithDepth(Ps, tic, ric);
                f_manager.triangulate(Ps, Rs, tic, ric); 
                solveOdometry();
                // solveMono(); 
                slideWindow(); 
                f_manager.removeFailures(); 
                last_R = Rs[WN]; 
                last_P = Ps[WN]; 
                last_R0 = Rs[0];
                last_P0 = Ps[0]; 
            }else{ // failed to initialize structure    
                ROS_DEBUG("RVIO.cpp: failed to initialize structure"); 
                slideWindow();
                cout<<"RVIO.cpp: after slideWindow() feature_manager has: "<<f_manager.feature.size()<<" features!"<<endl; 
            }
        }else{ // only wait for enough frame_count 
            frame_count++; 
        }
    }else{

        // only debug initialization 

        // f_manager.triangulateSimple(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulateWithDepth(Ps, tic, ric);
        f_manager.triangulate(Ps, Rs, tic, ric); 
        solveOdometry();
        // solveMono();

        slideWindow(); 
        f_manager.removeFailures(); 
        key_poses.clear(); 
        for(int i=0; i<=WN; i++)
            key_poses.push_back(Ps[i]); 

        last_R = Rs[WN]; 
        last_P = Ps[WN]; 
        last_R0 = Rs[0];
        last_P0 = Ps[0]; 
    }
    return ; 
}


void RVIO::solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

void RVIO::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void RVIO::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void RVIO::showStatus()
{
    cout<<"RVIO.cpp: showStatus: Poses: "<<endl; 
    for(int i=0; i<=frame_count; i++){
        cout << "Ps["<<i<<"]:"<< Ps[i].transpose()<<endl;
    }
    for(int i=0; i<=frame_count; i++){
        cout << "Vs["<<i<<"]:"<< Vs[i].transpose()<<endl;
    }
    for(int i=0; i<=frame_count; i++){
        cout << "Ba["<<i<<"]:"<< Bas[i].transpose()<<endl;
    }
    for(int i=0; i<=frame_count; i++){
        cout << "Bg["<<i<<"]:"<< Bgs[i].transpose()<<endl;
    }

}

void RVIO::slideWindow()
{
    if(marginalization_flag == MARGIN_OLD){

        double t_0 = Headers[0]; // .stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];

        if(frame_count == WN){
            for(int i=0; i<WN; i++){

                Headers[i] = Headers[i+1]; 
                Rs[i].swap(Rs[i+1]); 
                Ps[i].swap(Ps[i+1]); 

                std::swap(pre_integrations[i], pre_integrations[i+1]);
                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);

                bPls[i] = bPls[i+1]; 
                Pls[i].swap(Pls[i+1]);
            }
            Headers[WN] = Headers[WN-1]; 
            Ps[WN] = Ps[WN-1];
            Rs[WN] = Rs[WN-1];
            Vs[WN] = Vs[WN-1];
            Bas[WN] = Bas[WN-1]; 
            Bgs[WN] = Bgs[WN-1];
            bPls[WN] = bPls[WN-1]; 
            Pls[WN] = Pls[WN-1];

            delete pre_integrations[WN]; 
            pre_integrations[WN] = new IntegrationBase{acc_0, gyr_0, Bas[WN], Bgs[WN]};
            dt_buf[WN].clear();
            linear_acceleration_buf[WN].clear();
            angular_velocity_buf[WN].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }

            slideWindowOld(); 
        }else{
            cout<<"RVIO.cpp: what? in slide_window margin_old frame_count = "<<frame_count<<endl;
        }
    }else{

        if(frame_count == WN){

            Headers[WN-1] = Headers[WN]; 
            Ps[WN-1] = Ps[WN];
            Rs[WN-1] = Rs[WN];
            Vs[WN-1] = Vs[WN];
            Bas[WN-1] = Bas[WN]; 
            Bgs[WN-1] = Bgs[WN];
            bPls[WN-1] = bPls[WN]; 
            Pls[WN-1] = Pls[WN];

            for(int i=0; i<dt_buf[WN].size(); i++){
                double tmp_dt = dt_buf[WN][i]; 
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[WN][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[WN][i]; 
                pre_integrations[WN - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                dt_buf[WN - 1].push_back(tmp_dt);
                linear_acceleration_buf[WN - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[WN - 1].push_back(tmp_angular_velocity);
            }

            delete pre_integrations[WN]; 
            pre_integrations[WN] = new IntegrationBase{acc_0, gyr_0, Bas[WN], Bgs[WN]};
            dt_buf[WN].clear();
            linear_acceleration_buf[WN].clear();
            angular_velocity_buf[WN].clear();

            slideWindowNew();
        }else{
            cout<<"RVIO.cpp: what? in slide_window margin_new frame_count = "<<frame_count<<endl;
        }
    }
}

void RVIO::slideWindowNew()
{
    f_manager.removeFront(frame_count);
}

void RVIO::slideWindowOld()
{
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;

    if(shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }else
        f_manager.removeBack();
}
/*
void RVIO::solveMono()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalPrameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalPrameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    TicToc t_whole, t_prepare;
    // vector2double();
    priorOptimize(); 

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].pt;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.pt;
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                ceres::ResidualBlockId fid = problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                if(0){
                    static ofstream ouf("rgbd_vio_debug.txt"); 
                    vector<double*>* para = new vector<double*>;  
                    problem.GetParameterBlocksForResidualBlock(fid, para); 
                    vector<double> res(2); 
                    f->Evaluate(&para[0][0], &res[0], 0); 
                    // cout<<"estimator.cpp: residual: "<<res[0]<<" "<<res[1]<<endl;
                    ouf << f_m_cnt<<" "<<pts_i.transpose()<<" "<<pts_j.transpose()<<" "<<para_Feature[feature_index][0]<<" "<< res[0]<<" "<<res[1]<<endl;
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());
    // exit(0); 

    // double2vector();
    afterOptimize(); 

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // vector2double();
        priorOptimize(); 

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].pt;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.pt;
                 
                    
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            // vector2double();
            priorOptimize();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}
*/


void RVIO::solveMono()
{
    priorOptimize();
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WN + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalPrameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalPrameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WN; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WN - 2))
            continue;

        if(it_per_id.estimated_depth <= 0 || it_per_id.solve_flag == 2)
            continue; 
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].pt;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.pt;
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    
    afterOptimize();

    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        priorOptimize(); 

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WN - 2))
                    continue;

                if(it_per_id.estimated_depth <= 0 || it_per_id.solve_flag == 2)
                    continue; 

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].pt;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.pt;
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        marginalization_info->preMarginalize();      
        marginalization_info->marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            priorOptimize(); 
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            marginalization_info->preMarginalize();
            marginalization_info->marginalize();       
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
      
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
}


void RVIO::solveOdometry()
{
    priorOptimize();
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);    // it seems cauchyloss is much better than huberloss 
    // loss_function = new ceres::HuberLoss(1.0);
    
    assert(frame_count == WN);
    // ROS_DEBUG("RVIO.cpp: now frame_count = %d", frame_count);

    // add pose 
    for(int i=0; i<= frame_count; i++){
        ceres::LocalParameterization *local_param = new PoseLocalPrameterization(); 
        problem.AddParameterBlock(para_Pose[i], 7, local_param); 
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // fix the first pose 
    // problem.SetParameterBlockConstant(para_Pose[0]);
    for(int i=0; i<NUM_OF_CAM; i++){
        ceres::LocalParameterization *local_param = new PoseLocalPrameterization(); 
        problem.AddParameterBlock(para_Ex_Pose[i], 7, local_param); 
        // if not optimize [ric, tic]
        if(ESTIMATE_EXTRINSIC == 0)
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); 
    }


    //TODO: marginalization 
    if (last_marginalization_info && last_marginalization_info->valid){
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // for (int i = 0; i < last_marginalization_parameter_blocks.size(); ++i) {
        //     printf(" RVIO.cpp: last_marginalization_info parameter block %d pointer: %p \n", i, last_marginalization_parameter_blocks[i]); 
        // }
        problem.AddResidualBlock(marginalization_factor, NULL,
                               last_marginalization_parameter_blocks);
    }

    // add imu factor 
    for (int i = 0; i < frame_count; i++){
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0 )
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        // cout<<"IMU factor noise: "<<endl<<imu_factor->pre_integration->noise<<endl;
        // cout<<"IMU factor jacobian: "<<endl<<imu_factor->pre_integration->jacobian<<endl;
        // cout<<"IMU factor covariance: "<<endl<<imu_factor->pre_integration->covariance<<endl;
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    // add floor plane factor 
    for(int i=1; i<=frame_count; i++){
        if(bPls[i]){
            cout<<"RVIO.cpp: add plane factor "<<fp_Pls.transpose()<<" at pose i = "<<i<<" pl = "<<Pls[i].transpose()<<endl;
            PlaneFactor_P1 * plane_factor = new PlaneFactor_P1(fp_Pls, Pls[i]);
            problem.AddResidualBlock(plane_factor, NULL, para_Pose[i]); 
        }
    }

    int f_m_cnt = 0; 
    int feature_index = -1; 
    int cnt_used_features = 0; 

    for(auto &it_per_id : f_manager.feature){
        it_per_id.used_num = it_per_id.feature_per_frame.size(); 
        if(it_per_id.used_num < MIN_USED_NUM || it_per_id.start_frame >= WN - 2) continue; 
        ++cnt_used_features; 
        // feature with known depths
        if(it_per_id.estimated_depth > 0 && it_per_id.solve_flag != 2){

            ++feature_index; 
            if(feature_index >= NUM_OF_FEAT){
                ROS_ERROR("RVIO.cpp: feature_index = %d larger than %d ", feature_index, NUM_OF_FEAT); 
                continue; 
            }

            assert(it_per_id.depth_shift == 0); 
            int imu_i = it_per_id.start_frame + it_per_id.depth_shift; 
            Vector3d pts_i = it_per_id.feature_per_frame[it_per_id.depth_shift].pt; 

            for(int shift=0; shift<it_per_id.feature_per_frame.size(); shift++){
                double dpt_j = it_per_id.feature_per_frame[shift].dpt; 

                if(it_per_id.feature_per_frame[shift].lambda > 0 && it_per_id.feature_per_frame[shift].sig_l > 0){
                    dpt_j = 1./it_per_id.feature_per_frame[shift].lambda;
                }

                if(shift == it_per_id.depth_shift) {
                    if(dpt_j > 0 && dpt_j <= DPT_VALID_RANGE){
                        // if(dpt_j <= 2.2){
                        //     problem.SetParameterBlockConstant(para_Feature[feature_index]); 
                        //}else{
                        if(0){  // add single depth constraint 
                            SingleInvDepthFactor* fs = new SingleInvDepthFactor(1./dpt_j); 
                            if(it_per_id.feature_per_frame[shift].lambda > 0 && it_per_id.feature_per_frame[shift].sig_l > 0)
                                fs->setSigma(it_per_id.feature_per_frame[shift].sig_l);
                            problem.AddResidualBlock(fs, loss_function, para_Feature[feature_index]);
                        }

                        // TODO: add covariance info
                        if(dpt_j <= DPT_VALID_RANGE){
                            // problem.SetParameterBlockConstant(para_Feature[feature_index]); 
                        }
                        f_m_cnt++;
                    }
                    continue;
                } 

                int imu_j = it_per_id.start_frame + shift; 
                Vector3d pts_j = it_per_id.feature_per_frame[shift].pt;                

                if(dpt_j <= 0 ){
                    // para_Feature[feature_index][0] = 1./it_per_id.estimated_depth; 
                    ProjectionFactor * f = new ProjectionFactor(pts_i, pts_j); 
                    // f->sqrt_info = 240 * Eigen::Matrix2d::Identity(); // 240
                    // SampsonFactorCross *f = new SampsonFactorCross(pts_i, pts_j); 
                    // cout <<" RVIO.cpp: factor between: "<<imu_i<<" "<<imu_j<<" pts_i: "<<pts_i.transpose()<<" pts_j: "<<pts_j.transpose()<<" depth: "<<para_Feature[feature_index][0]<<endl;
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                    // if(pt.v == ip_M::DEPTH_MES)
                }else{
                    ProjectionDepthFactor * f = new ProjectionDepthFactor(pts_i, pts_j, 1./dpt_j);
                    if(it_per_id.feature_per_frame[shift].lambda > 0 && it_per_id.feature_per_frame[shift].sig_l > 0){
                        Eigen::Matrix3d C = Eigen::Matrix3d::Identity()*(1.5/FOCAL_LENGTH); 
                        C(2,2) = it_per_id.feature_per_frame[shift].sig_l; 
                        f->setSqrtCov(C);
                    }
                    ceres::ResidualBlockId fid = problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                    /*if(1){
                        // static ofstream ouf("rgbd_vio_debug.txt"); 
                        vector<double*>* para = new vector<double*>;  
                        problem.GetParameterBlocksForResidualBlock(fid, para); 
                        vector<double> res(3); 
                        f->Evaluate(&para[0][0], &res[0], 0);
                        if(res[0] != res[0]) {
                            ROS_ERROR("residual is NaN ");
                            cout << pts_i.transpose()<<" "<<pts_j.transpose()<<" "<<para_Feature[feature_index][0]<<" "<< res[0]<<" "<<res[1]<<endl;
                        }
                        // cout<<"estimator.cpp: residual: "<<res[0]<<" "<<res[1]<<endl;
                        // ouf << f_m_cnt<<" "<<pts_i.transpose()<<" "<<pts_j.transpose()<<" "<<para_Feature[feature_index][0]<<" "<< res[0]<<" "<<res[1]<<endl;
                    }*/
                }
                f_m_cnt++;
            }
            // if(it_per_id.dpt_type == DEPTH_MES)
            //    problem.SetParameterBlockConstant(para_Feature[feature_index]);   

        }else if(it_per_id.solve_flag != 2){ // feature unknown depths 

            int imu_i = it_per_id.start_frame; 
            Vector3d pts_i = it_per_id.feature_per_frame[0].pt; 

            for(int shift = 1; shift < it_per_id.feature_per_frame.size(); shift++){

                int imu_j = imu_i + shift; 
                Vector3d pts_j = it_per_id.feature_per_frame[shift].pt; 

                ProjectionFactor_Y2 * f= new ProjectionFactor_Y2(pts_i, pts_j); 
                // SampsonFactorEssential * f = new SampsonFactorEssential(pts_i, pts_j); 
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0]); 
            }
        }
    }

    ROS_DEBUG("RVIO.cpp: before optimization, %d features have been used with %d constrints!", cnt_used_features, f_m_cnt);
    // optimize it 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    options.max_solver_time_in_seconds = SOLVER_TIME; 
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    afterOptimize();
    
    // TODO: add marginalization 
    if(marginalization_flag == MARGIN_OLD){
        MarginalizationInfo * marginalization_info = new MarginalizationInfo(); 

        priorOptimize(); 

        if(last_marginalization_info && last_marginalization_info->valid){
            vector<int> drop_set;
            for(int i=0; i<last_marginalization_parameter_blocks.size(); i++){
                if(last_marginalization_parameter_blocks[i] == para_Pose[0] || 
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) 
                        drop_set.push_back(i); 
            } 

            // construct new marginalization factor 
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info); 
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set); 
            // for (int i = 0; i < residual_block_info->parameter_blocks.size(); ++i) {
            //     printf(" RVIO.cpp: MarginalizationFactor parameter block %d pointer: %p \n", i, residual_block_info->parameter_blocks[i]); 
            // }
            marginalization_info->addResidualBlockInfo(residual_block_info); 
        }

        // for imu 
        if(pre_integrations[1]->sum_dt < 10.0){
            IMUFactor * imu_factor = new IMUFactor(pre_integrations[1]); 
            ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(imu_factor, NULL, vector<double*>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                       vector<int>{0,1});
            // for (int i = 0; i < residual_block_info->parameter_blocks.size(); ++i) {
            //     printf(" RVIO.cpp: IMUFactor parameter block %d pointer: %p \n", i, residual_block_info->parameter_blocks[i]); 
            // }
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        
        // for features 
        int feature_index = -1; 
        for(auto& it_per_id : f_manager.feature){
            it_per_id.used_num = it_per_id.feature_per_frame.size(); 
            if(it_per_id.used_num < MIN_USED_NUM || it_per_id.start_frame >= WN - 2) 
                continue; // no constraint for this feature 

        if(it_per_id.estimated_depth > 0 && it_per_id.solve_flag != 2){

            ++feature_index; 
            if(feature_index >= NUM_OF_FEAT){
                ROS_ERROR("RVIO.cpp: feature_index = %d larger than %d ", feature_index, NUM_OF_FEAT); 
                continue; 
            }

            if(it_per_id.start_frame != 0) 
                continue; // no worry 

            int imu_i = it_per_id.start_frame + it_per_id.depth_shift;
            Vector3d pts_i = it_per_id.feature_per_frame[it_per_id.depth_shift].pt; 
            
            if(imu_i == 0){ // marginalized the node with depth 
                for(int imu_j=0; imu_j<it_per_id.feature_per_frame.size(); imu_j++){

                    if(imu_j == imu_i){
                        // if(dpt_j > 0){
                        //     // add single depth constraint 
                        //     SingleInvDepthFactor* fs = new SingleInvDepthFactor(1./dpt_j); 
                        //     // problem.AddResidualBlock(fs, loss_function, para_Feature[feature_index]);
                        //     ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(f, loss_function, 
                        //                     vector<double*>{para_Feature[feature_index]},
                        //                     vector<int>{0});
                        // }
                        continue ; 
                    }

                    Vector3d pts_j = it_per_id.feature_per_frame[imu_j].pt;
                    double dpt_j = it_per_id.feature_per_frame[imu_j].dpt; 
                    if(it_per_id.feature_per_frame[imu_j].lambda > 0 && it_per_id.feature_per_frame[imu_j].sig_l > 0){
                        dpt_j = 1./it_per_id.feature_per_frame[imu_j].lambda; 
                    }

                    // TODO: figure out why marginalization with depth variables will decrease 
                    if(1 || dpt_j <= 0){
                        // para_Feature[feature_index][0] = 1./it_per_id.estimated_depth; 
                        ProjectionFactor * f = new ProjectionFactor(pts_i, pts_j); 

                        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(f, loss_function, 
                                                            vector<double*>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                            vector<int>{0, 3}); 
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }else{
                        ProjectionDepthFactor * f = new ProjectionDepthFactor(pts_i, pts_j, 1./dpt_j);
                        if(it_per_id.feature_per_frame[imu_j].lambda > 0 && it_per_id.feature_per_frame[imu_j].sig_l > 0){
                            Eigen::Matrix3d C = Eigen::Matrix3d::Identity()*(1.5/FOCAL_LENGTH); 
                            C(2,2) = it_per_id.feature_per_frame[imu_j].sig_l; 
                            f->setSqrtCov(C);
                        }

                        ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(f, loss_function, 
                                                            vector<double*>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                            vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                        // string blocks;
                        // for (int i = 0; i < residual_block_info->parameter_blocks.size(); ++i) {
                            // printf(" RVIO.cpp: up parameter block %d pointer: %p \n", i, residual_block_info->parameter_blocks[i]); 
                        // }
                    }
                }
            }else{
                ROS_ERROR("now should not arrive here!");
            }

        }else if(it_per_id.solve_flag != 2){ // feature unknown depths 

            // don't handle 
        }

        }

        // no need to worry floor plane right now 
        TicToc t_pre_margin; 
        marginalization_info->preMarginalize(); 
        ROS_DEBUG("RVIO.cpp: pre marginalization: %f ms", t_pre_margin.toc()); 

        TicToc t_margin;
        marginalization_info->marginalize(); 
        ROS_DEBUG("RVIO.cpp: marginalization %f ms", t_margin.toc()); 

        std::unordered_map<long, double*> addr_shift; 
        for(int i=1; i<= WN; i++){
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i-1]; 
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i-1]; 
        }
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[0])] = para_Ex_Pose[0]; 
        // addr_shift[reinterpret_cast<long>(para_Ex_Pose[1])] = para_Ex_Pose[1]; 
        vector<double*> param_blocks = marginalization_info->getParameterBlocks(addr_shift); 
        if(last_marginalization_info) 
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks =param_blocks;

    }else{
        
        if(last_marginalization_info && 
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WN-1])){

            MarginalizationInfo * marginalization_info = new MarginalizationInfo(); 
            priorOptimize(); 
            if(last_marginalization_info && last_marginalization_info->valid){
                vector<int> drop_set; 
                for(int i=0; i<last_marginalization_parameter_blocks.size(); i++){
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WN-1]); 
                    if(last_marginalization_parameter_blocks[i] == para_Pose[WN-1])
                        drop_set.push_back(i); 
                }

                MarginalizationFactor* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL, 
                                                            last_marginalization_parameter_blocks, drop_set); 
                marginalization_info->addResidualBlockInfo(residual_block_info);

            }

            
            // no need to worry floor plane right now 
            TicToc t_pre_margin; 
            marginalization_info->preMarginalize(); 
            ROS_DEBUG("RVIO.cpp: pre marginalization: %f ms", t_pre_margin.toc()); 

            TicToc t_margin;
            marginalization_info->marginalize(); 
            ROS_DEBUG("RVIO.cpp: marginalization %f ms", t_margin.toc()); 


            std::unordered_map<long, double*> addr_shift; 
            for(int i=0; i<= WN; i++){
                if(i == WN-1) continue; 
                else if(i== WN){
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i-1]; 
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i-1]; 
                }else{
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i]; 
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i]; 
                }
            }
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[0])] = para_Ex_Pose[0]; 
            // addr_shift[reinterpret_cast<long>(para_Ex_Pose[1])] = para_Ex_Pose[1]; 
            vector<double*> param_blocks = marginalization_info->getParameterBlocks(addr_shift); 
            if(last_marginalization_info) 
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks =param_blocks;
        }
    }

    return ; 
}


void RVIO::afterOptimize()
{
    assert(frame_count == WN);

    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

     // handle pose 
    for(int i=0; i<=frame_count; i++)
    {
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);
        // cout <<" in after: para_Speed: "<<para_SpeedBias[i][0]<<" "<<para_SpeedBias[i][1]<<" "<<para_SpeedBias[i][2]<<endl;
        // cout << " in after rot_diff: "<<endl<<rot_diff<<endl;
        // cout << "in after: V["<<i<<"] : "<<Vs[i].transpose()<<endl;

        Bas[i] = Vector3d(para_SpeedBias[i][3],
        para_SpeedBias[i][4],
        para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
        para_SpeedBias[i][7],
        para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
	tic[i] = Vector3d(para_Ex_Pose[i][0],
		para_Ex_Pose[i][1],
		para_Ex_Pose[i][2]);
	ric[i] = Quaterniond(para_Ex_Pose[i][6],
		para_Ex_Pose[i][3],
		para_Ex_Pose[i][4],
		para_Ex_Pose[i][5]).toRotationMatrix();
    }
    Eigen::Quaterniond q(Rs[frame_count]);
    mCurrIMUPose = tf::Transform(tf::Quaternion(q.x(), q.y(), q.z(), q.w()), tf::Vector3(Ps[frame_count][0], Ps[frame_count][1], Ps[frame_count][2]));
    ROS_INFO("RVIO.cpp: afterOptimize at frame_count = %d", frame_count);
    // printTF(mCurrIMUPose, "RVIO.cpp: after optimization mCurrIMUPose: ");

    q = Eigen::Quaterniond(ric[0]); 
    mTIC = tf::Transform(tf::Quaternion(q.x(), q.y(), q.z(), q.w()), tf::Vector3(tic[0][0], tic[0][1], tic[0][2]));

    mCurrPose = mCurrIMUPose * mTIC; 

   // for(int i=0; i<vip.size() && i< NUM_OF_FEAT; i++)
    int index = 0; 
    for(auto& it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WN - 2){
            if(it_per_id.estimated_depth > 0 && it_per_id.solve_flag != 2)
                // para_Feature[index++][0] = 1./it_per_id.estimated_depth; 
                it_per_id.estimated_depth = 1./ para_Feature[index++][0];
                if(it_per_id.estimated_depth <= 0.1 )
                    it_per_id.solve_flag = 2;
                else
                    it_per_id.solve_flag = 1;
        }
        if(index >= NUM_OF_FEAT) break;
    }
    /*for(int i=0; i<vip.size(); i++)
    {
	ip_M& m = vip[i];
	if(m.v == ip_M::DEPTH_TRI)
	{
	    m.s = 1./para_Feature[i][0];
	}
    }*/
    return ; 
}

void RVIO::priorOptimize()
{

    assert(frame_count == WN); 

    // handle pose 
    for(int i=0; i<=frame_count; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
        // cout << "in prior: V["<<i<<"] : "<<Vs[i].transpose()<<endl;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
    	para_Ex_Pose[i][0] = tic[i].x();
    	para_Ex_Pose[i][1] = tic[i].y();
    	para_Ex_Pose[i][2] = tic[i].z();
    	Quaterniond q{ric[i]};
    	para_Ex_Pose[i][3] = q.x();
    	para_Ex_Pose[i][4] = q.y();
    	para_Ex_Pose[i][5] = q.z();
    	para_Ex_Pose[i][6] = q.w();
    }
    
    // for(int i=0; i<vip.size() && i< NUM_OF_FEAT; i++)
    int index = 0; 
    for(auto& it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // if(it_per_id.used_num >= MIN_USED_NUM){
        if(it_per_id.used_num >= MIN_USED_NUM && it_per_id.start_frame < WN - 2){
            if(it_per_id.estimated_depth > 0 && it_per_id.solve_flag != 2) // 2: means failed to track 
                para_Feature[index++][0] = 1./it_per_id.estimated_depth; 
                // it_per_id.estimated_depth = 1./ para_Feature[index++][0];
        }
        if(index >= NUM_OF_FEAT) break;
    }

    // TODO: use it later 
    para_Td[0][0] = 0; 
}

