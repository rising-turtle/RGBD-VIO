/*
    Feb. 13 2020, He Zhang, hzhang8@vcu.edu 
    
    RVIO: handle plane related functions 

*/

#include "rvio.h"
#include "plane.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr RVIO::processDepthImage(const cv::Mat& dpt_img)
{
    // median filter to get rid some noise 
    // cv::Mat dpt_img = cv_bridge::toCvCopy(dpt_img)->image;
    // cv::Mat dst; 
    // cv::medianBlur(dpt_img, dst, 5);  
    // dpt_img = dst; 

    pcl::PointCloud<pcl::PointXYZ>::Ptr tmpPC(new pcl::PointCloud<pcl::PointXYZ>); 
    double cloud_dense_rate = 5; 
    double halfDS = cloud_dense_rate/2. - 0.5; 
    float scale = 0.001; 
    float min_dis = 0.3; 
    float max_dis = 7.0; // mMaxDepth;  // keep depth range 
    for(double i = halfDS; i < dpt_img.rows; i += cloud_dense_rate)
    for(double j = halfDS; j < dpt_img.cols; j += cloud_dense_rate)
    {
        int pixelCnt = 0; 
        float vd, vd_sum = 0; 
        int is = (int)(i - halfDS); int ie = (int)(i + halfDS); 
        int js = (int)(j - halfDS); int je = (int)(j + halfDS);
        for(int ii = is; ii<= ie; ii++)
        for(int jj = js; jj<= je; jj++)
        {
            unsigned short _dpt = dpt_img.at<unsigned short>(ii, jj); 
            vd = _dpt * scale; 
            // vd = syncCloud2Pointer[ii * dpt_img.cols + jj]; 
            if(vd > min_dis && vd < max_dis)
            {
            pixelCnt++; 
            vd_sum += vd; 
            }
        }
        if(pixelCnt > 0)
        {
            double u = (j - CX)/FX;
            double v = (i - CY)/FY; 
            double mean_vd = vd_sum / pixelCnt; 
            pcl::PointXYZ pt;
            pt.x = u * mean_vd; 
            pt.y = v * mean_vd;
            pt.z = mean_vd; 
            // pt.intensity = 1; // timeElapsed;
            tmpPC->points.push_back(pt); 
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointer(new pcl::PointCloud<pcl::PointXYZ>); 
    pcl::VoxelGrid<pcl::PointXYZ> downSizeFilter;
    downSizeFilter.setInputCloud(tmpPC);
    downSizeFilter.setLeafSize(0.03, 0.03, 0.03);
    downSizeFilter.filter(*cloudPointer);
    return cloudPointer; 
}

// extract floor and 
bool RVIO::getFloorAndObstacle(const cv::Mat& dpt)
{
	TicToc t_fz;
	bool ret = false; 
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc = processDepthImage(dpt);

	if(in_pc->points.size() < 100)
		return false; 

	// transform into global coordinate system 
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_pc_w(new pcl::PointCloud<pcl::PointXYZ>); 
	for(int i=0; i<in_pc->points.size(); i++){

		pcl::PointXYZ& pti = in_pc->points[i]; 
		tf::Vector3 pii(pti.x, pti.y, pti.z); 
	    tf::Vector3 pww = mCurrPose * pii; // transform from F[frame_count] to world coordinate system  
	    pcl::PointXYZ ptw(pww.getX(), pww.getY(), pww.getZ());
	    in_pc_w->points.push_back(ptw); 
	}
	in_pc_w->width = in_pc_w->points.size(); 
	in_pc_w->height = 1; 

    double min_z, max_z; 

    // if(mFloorZ == NOT_INITIED) //     
    if(mbFirstFloorObserved== false) 
    {	
	// double max_z = 0.1; 
	// double min_z = -1.5; 
	   min_z = 1e10; 
	   max_z = -1e10; 
        for(int i=0; i<in_pc_w->points.size(); i++)
        {
            double pz = in_pc_w->points[i].z; 
            if(min_z > pz ) min_z = pz; 
            if(max_z < pz ) max_z = pz; 
        }
    }
    else{
        min_z = mFloorZ - 3*mFloorRange; 
        max_z = mFloorZ + 3*mFloorRange; 
    }

	// cout <<"removeFloorPts.cpp : min_z: "<<min_z<<" max_z: "<<max_z<<endl; 

	//  histogram into different bins 
	double res = 0.1; 
	if(max_z - min_z > 2.0) max_z = min_z + 2.0; 
	int n_bins = (max_z - min_z)/res + 1; 

	map<int, vector<int> > bins; 
	for(int i=0; i<in_pc_w->points.size(); i++)
	{
	    double pz = in_pc_w->points[i].z; 
	    int ind = (pz - min_z + res/2.)/res;
	    bins[ind].push_back(i); 
	}

	// find the bin with most points 
	int max_n = 0; 
	int max_id = -1; 
	map<int, vector<int> >::iterator it = bins.begin();
	while(it != bins.end())
	{
	    if(it->second.size() > max_n) 
	    {
		  max_n = it->second.size(); 
		  max_id = it->first; 
	    }
	    ++it; 
	}
    if(mFloorZ == NOT_INITIED)
	   mFloorZ = min_z + max_id*res; 
    else {
        if(max_n > 100)
            mFloorZ = 0.5 * mFloorZ + 0.5 *(min_z + max_id*res);
    }

    // find floor plane 
    mPCFloor->clear(); 
    mPCNoFloor->points.clear(); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());
    tmp->points.reserve(in_pc_w->points.size()/2); 
    for(int i=0; i<in_pc_w->points.size(); i++)
    {

        double pz = in_pc_w->points[i].z; 

        if(mbFirstFloorObserved == false){
            double px = in_pc_w->points[i].x; 
            double py = in_pc_w->points[i].y; 
            double dd = sqrt(px*px + py*py + pz*pz); 
            if(dd > 2.4) continue;
        }

        if(pz < mFloorZ + mFloorRange && pz > mFloorZ - mFloorRange)
        {
          tmp->points.push_back(in_pc_w->points[i]); 
        }
    }
    if(tmp->points.size() < 150)
        return false; 
    Eigen::Vector3d nv; 
    double nd; 
    pcl::PointIndices::Ptr indices(new pcl::PointIndices); 
    ((Plane*)(0))->computeByPCL<pcl::PointXYZ>(tmp, indices, nv, nd, 0.1); 
    Eigen::Vector3d g(fp_Pls[0], fp_Pls[1], fp_Pls[2]); 
    double angle = nv.dot(g); 
    if(angle < 0){
    	nv = nv * -1.; 
    	angle *= -1; 
    }
    const double COS30 = cos(30.*M_PI/180.);
    const double COS10 = cos(10.*M_PI/180.);
    const double COS5 = cos(5.*M_PI/180.);
    // cout<<"Floor plane has indices.size = "<<indices->indices.size()<<" points nv = "<<nv.transpose()<<endl;
    if(indices->indices.size() < 500 || angle < COS5) // NO Floor plane is observed 
    // if(indices->indices.size() < 100 || angle < COS10) // NO Floor plane is observed 
    // if(indices->indices.size() < 100 || angle < COS30) // NO Floor plane is observed 
    {
        mPCNoFloor->points.reserve(in_pc_w->points.size()); 
        for(int i=0; i<in_pc_w->points.size(); i++)
        {
            double pz = in_pc_w->points[i].z; 
            if(pz > mFloorZ + 3*mFloorRange)
            {
		        // out->points.push_back(in->points[i]); 
				pcl::PointXYZI pt; 
				pt.x = in_pc_w->points[i].x; 
				pt.y = in_pc_w->points[i].y; 
				pt.z = in_pc_w->points[i].z; 
				pt.intensity = pz - mFloorZ; // intensity contains distance to floor plane 
				mPCNoFloor->points.push_back(pt); 
            }
        }

        // no floor detected at this keyframe 
        bPls[frame_count] = false; 

    }else // succeed to get a floor plane 
    { 
    	ret = true; 
        if(mbFirstFloorObserved == false)
        {
            mbFirstFloorObserved = true; 
            fp_Pls[0] = nv(0); // nv(0); // 0; // Pls[0][0] = 0; 
            fp_Pls[1] = nv(1); // nv(1); // 0; // Pls[0][1] = 0; 
            fp_Pls[2] = nv(2); // nv(2); // 1.; // Pls[0][2] = 1.;
            fp_Pls[3] = nd; // Pls[0][3] = nd;
        }
        // save floor points for display 
        mPCFloor->points.reserve(indices->indices.size()); 
        double sum_z = 0; 
        for(int i=0; i<indices->indices.size(); i++)
        {
            mPCFloor->points.push_back(tmp->points[indices->indices[i]]); 
            sum_z += tmp->points[indices->indices[i]].z; 
        }
        mFloorZ = sum_z/(indices->indices.size()); 
        cout <<"rvio_plane.cpp: succeed to find out floor plane, reset floor_Z = "<<mFloorZ<<endl;

        mPCFloor->width = mPCFloor->points.size(); 
        mPCFloor->height = 1;
        mPCFloor->is_dense = true;
        // find the obstacle point 
        for(int i=0; i<in_pc_w->points.size(); i++)
        {
            Eigen::Vector3d pt(in_pc_w->points[i].x, in_pc_w->points[i].y, in_pc_w->points[i].z);
            double dis = nv.dot(pt) + nd; 
            if(dis > 2*mFloorRange)
            {
                // out->points.push_back(in->points[i]);
				pcl::PointXYZI pt; 
				pt.x = in_pc_w->points[i].x; 
				pt.y = in_pc_w->points[i].y; 
				pt.z = in_pc_w->points[i].z; 
				pt.intensity = dis; // intensity contains distance to floor plane 
				mPCNoFloor->points.push_back(pt); 
            }
        }
    
    	// floor plane detected at this keyframe 
    	bPls[frame_count] = true; 
    	// now transfer it into current IMU frame 
	    {
			tf::Quaternion tq = mCurrIMUPose.getRotation(); 
			tf::Vector3 tt = mCurrIMUPose.getOrigin(); 
			
			Eigen::Quaterniond q(tq.getW(), tq.getX(), tq.getY(), tq.getZ()); 
			Eigen::Vector3d t(tt.getX(), tt.getY(), tt.getZ()); 
			
			Eigen::Vector3d nl = q.inverse() * nv; 
			double dl = nv.dot(t) + nd; 
			Pls[frame_count][0] = nl(0); 
    		Pls[frame_count][1] = nl(1); 
    		Pls[frame_count][2] = nl(2); 
    		Pls[frame_count][3] = dl;
	    }
    }
  	
  	// set non floor point 
  	mPCNoFloor->width = mPCNoFloor->points.size(); 
    mPCNoFloor->height = 1; 
   	mPCNoFloor->is_dense = true; 
 
    // cout <<"DVIO.cpp: outpc has "<<out->points.size()<<" points floor_z = "<<mFloorZ<<endl;
    ROS_DEBUG("rvio_plane.cpp: remove floor cost %f ms", t_fz.toc()); 
    return true ;
}

Eigen::Quaterniond RVIO::rotateToG(Eigen::Vector3d& fv)
{
      // compute rotation for the first pose 
    // Eigen::Vector3d fv(ax, ay, az); 
    Eigen::Vector3d tv(0, 0, 1);  // vn100's gz points to upwards
    Eigen::Vector3d w = fv.cross(tv).normalized(); 
    double angle = acos(fv.dot(tv)); 
    
    double half_angle = angle /2.;
    Eigen::Vector4d vq; 
    vq.head<3>() = w * sin(half_angle); 
    vq[3] = cos(half_angle); 

    // cout <<"w = "<<w.transpose()<<" angle = "<<angle<<" vq = "<<vq.transpose()<<endl; 
    Eigen::Quaterniond q(vq); 
    return q;
}
