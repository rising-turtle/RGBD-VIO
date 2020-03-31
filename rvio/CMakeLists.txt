project(rgbd_vio)

include_directories("./initial" "./utility")

add_library(rgbd_vio SHARED parameters.cpp marginalization_factor.cpp projection_quat.cpp rvio.cpp rvio_init.cpp rvio_plane.cpp rvio_dvio.cpp depth_factor.cpp
				 visualization.cpp feature_manager.cpp initial/gmm_model.cpp initial/initial_alignment.cpp initial/initial_sfm.cpp 
				 initial/solve_5pts.cpp initial/initial_ex_rotation.cpp initial/translate_factor.cpp initial/solve_opt.cpp ./utility/utility.cpp)


target_link_libraries(rgbd_vio ${CERES_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(rgbd_vio_node rvio_syn_node.cpp)
target_link_libraries(rgbd_vio_node rgbd_vio ${CERES_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

# add_executable(dvio_init_node dvio_init_node.cpp)
# target_link_libraries(dvio_init_node rvio ${CERES_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(feature_tracker_rgbd_vio feature_tracker/feature_tracker.cpp feature_tracker/feature_tracker_node.cpp parameters.cpp)
target_link_libraries(feature_tracker_rgbd_vio ${CERES_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})


