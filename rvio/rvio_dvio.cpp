#include "rvio.h"
#include "../utility/utility.h"
#include "opencv/cv.h"
#include <iostream>
#include <string>
#include <thread>
#include "../utility/tic_toc.h"
#include "projection_quat.h"
#include "visualization.h"
#include "depth_factor.h"

using namespace QUATERNION_VIO; 
using namespace Eigen;

 // functions for dvio 
 void RVIO::processImage_Init_dvio(const map<int, vector<pair<int, Eigen::Matrix<double, 10, 1>>>> &image, const double header)
 {
 	ROS_DEBUG("timestamp %lf with feature points %lu", header, image.size());

    // if(f_manager.addFeatureCheckParallaxSigma(frame_count, image))
    if(f_manager.addFeatureCheckParallaxSigma_dvio(frame_count, image, (solver_flag!=INITIAL)))
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
        cout<<"RVIO.cpp: at frame_count: "<<frame_count<<" feature_manager has: "<<f_manager.feature.size()<<" features!"<<endl; 
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

                f_manager.triangulateWithDepth(Ps, tic, ric);
                // f_manager.triangulate(Ps, Rs, tic, ric); 
                // f_manager.triangulateSimple(frame_count, Ps, Rs, tic, ric);
                // solveOdometry();
                // solveMono();
                solveOdometry_dvio();  
                slideWindow_dvio(); 
                // slideWindow();
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
        // f_manager.triangulateSimple(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulateWithDepth(Ps, tic, ric);
        // f_manager.triangulate(Ps, Rs, tic, ric); 
        // solveOdometry();
        // solveMono();
        solveOdometry_dvio(); 
        slideWindow_dvio();
        // slideWindow(); 
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

void RVIO::slideWindow_dvio()
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
            bPls[WN] = false; 

            delete pre_integrations[WN]; 
            pre_integrations[WN] = new IntegrationBase{acc_0, gyr_0, Bas[WN], Bgs[WN]};
            dt_buf[WN].clear();
            linear_acceleration_buf[WN].clear();
            angular_velocity_buf[WN].clear();

            slideWindowOld_dvio(); 
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

            slideWindowNew_dvio();
        }else{
            cout<<"rvio_dvio.cpp: what? in slide_window margin_new frame_count = "<<frame_count<<endl;
        }
    }
}

void RVIO::slideWindowNew_dvio()
{
    // f_manager.removeFront(frame_count);
    f_manager.removeFrontWithDepth(frame_count);
}

void RVIO::slideWindowOld_dvio()
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
        f_manager.removeBackShiftDepth_dvio(R0, P0, R1, P1);
    }else
        f_manager.removeBack();
}

void RVIO::solveOdometry_dvio()
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
        problem.AddResidualBlock(marginalization_factor, NULL,
                               last_marginalization_parameter_blocks);
    }

    // add imu factor 
    for (int i = 0; i < frame_count; i++){
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0 )
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    // add floor plane factor 
    for(int i=1; i<=frame_count; i++){
    // for(int i=frame_count-2; i<=frame_count; i++){
        if(bPls[i]){
            cout<<"rvio_dvio.cpp: add plane factor "<<fp_Pls.transpose()<<" at pose i = "<<i<<" pl = "<<Pls[i].transpose()<<endl;
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
                ROS_ERROR("rvio_dvio.cpp: feature_index = %d larger than %d ", feature_index, NUM_OF_FEAT); 
                continue; 
            }

            int imu_i = it_per_id.start_frame + it_per_id.depth_shift; 
            Vector3d pts_i = it_per_id.feature_per_frame[it_per_id.depth_shift].pt; 

            for(int shift=0; shift<it_per_id.feature_per_frame.size(); shift++){
                double dpt_j = it_per_id.feature_per_frame[shift].dpt; 

                if(it_per_id.feature_per_frame[shift].lambda > 0 && it_per_id.feature_per_frame[shift].sig_l > 0){
                    dpt_j = 1./it_per_id.feature_per_frame[shift].lambda;
                }

                if(shift == it_per_id.depth_shift) {
                    continue;
                } 

                int imu_j = it_per_id.start_frame + shift; 
                Vector3d pts_j = it_per_id.feature_per_frame[shift].pt;                
           		
           		if(1 || dpt_j <= 0 ){ // do not consider the
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
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                }           
                f_m_cnt++;
            }
            if(it_per_id.dpt_type == DEPTH_MES && it_per_id.estimated_depth < DPT_VALID_RANGE)
                problem.SetParameterBlockConstant(para_Feature[feature_index]);   

        }else if(it_per_id.solve_flag != 2){ // feature unknown depths 

            int imu_i = it_per_id.start_frame; 
            Vector3d pts_i = it_per_id.feature_per_frame[0].pt; 

            for(int shift = 1; shift < it_per_id.feature_per_frame.size(); shift++){

                int imu_j = imu_i + shift; 
                Vector3d pts_j = it_per_id.feature_per_frame[shift].pt; 

                ProjectionFactor_Y2 * f= new ProjectionFactor_Y2(pts_i, pts_j); 
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0]); 
            }
        }
    }

    ROS_DEBUG("rvio_dvio.cpp: before optimization, %d features have been used with %d constrints!", cnt_used_features, f_m_cnt);
    // optimize it 
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = SOLVER_TIME; 
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("rvio_dvio.cpp: Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("rvio_dvio.cpp: solver costs: %f", t_solver.toc());

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
            marginalization_info->addResidualBlockInfo(residual_block_info); 
        }

        // for imu 
        if(pre_integrations[1]->sum_dt < 10.0){
            IMUFactor * imu_factor = new IMUFactor(pre_integrations[1]); 
            ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(imu_factor, NULL, vector<double*>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                       vector<int>{0,1});
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
                ROS_ERROR("rvio_dvio.cpp: feature_index = %d larger than %d ", feature_index, NUM_OF_FEAT); 
                continue; 
            }

            if(it_per_id.start_frame != 0) 
                continue; // no worry 

            int imu_i = it_per_id.start_frame + it_per_id.depth_shift;
            Vector3d pts_i = it_per_id.feature_per_frame[it_per_id.depth_shift].pt; 
            
            if(imu_i == 0){ // marginalized the node with depth 
                for(int imu_j=0; imu_j<it_per_id.feature_per_frame.size(); imu_j++){

                    if(imu_j == imu_i){
                        continue ; 
                    }

                    Vector3d pts_j = it_per_id.feature_per_frame[imu_j].pt;
                    double dpt_j = it_per_id.feature_per_frame[imu_j].dpt; 
                    if(it_per_id.feature_per_frame[imu_j].lambda > 0 && it_per_id.feature_per_frame[imu_j].sig_l > 0){
                        dpt_j = 1./it_per_id.feature_per_frame[imu_j].lambda; 
                    }

     
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
            	// ROS_ERROR("now should not arrive here!");
                // Vector3d pts_j = it_per_id.feature_per_frame[0].pt;
                // double dpt_j = it_per_id.feature_per_frame[0].dpt; 
                // if(it_per_id.feature_per_frame[0].lambda > 0 && it_per_id.feature_per_frame[0].sig_l > 0){
                //     dpt_j = 1./it_per_id.feature_per_frame[0].lambda; 
                // }
                // ProjectionFactor * f = new ProjectionFactor(pts_i, pts_j); 
                // ResidualBlockInfo* residual_block_info = new ResidualBlockInfo(f, loss_function, 
                //                                     vector<double*>{para_Pose[imu_i], para_Pose[0], para_Ex_Pose[0], para_Feature[feature_index]},
                //                                     vector<int>{1, 3});
                // marginalization_info->addResidualBlockInfo(residual_block_info);   
            }

        }else if(it_per_id.solve_flag != 2){ // feature unknown depths 
            // do nothing 
        }

        }

        // no need to worry floor plane right now 
        TicToc t_pre_margin; 
        marginalization_info->preMarginalize(); 
        ROS_DEBUG("rvio_dvio.cpp: pre marginalization: %f ms", t_pre_margin.toc()); 

        TicToc t_margin;
        marginalization_info->marginalize(); 
        ROS_DEBUG("rvio_dvio.cpp: marginalization %f ms", t_margin.toc()); 

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
            ROS_DEBUG("rvio_dvio.cpp: pre marginalization: %f ms", t_pre_margin.toc()); 

            TicToc t_margin;
            marginalization_info->marginalize(); 
            ROS_DEBUG("rvio_dvio.cpp: marginalization %f ms", t_margin.toc()); 


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
