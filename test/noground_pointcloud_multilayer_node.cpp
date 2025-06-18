#include <iostream>
#include <memory>
#include <cmath>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include "MultiLayerSOGMMap.hpp"
#include "patchworkpp/patchworkpp.hpp"

#include <multilayer/VoxelGridMsg.h>
#include <multilayer/VoxelGridMsgArray.h>

// 相机的内参、位姿信息、深度图像和点云数据
struct cameraData
{
    /* depth image process */
    double cx, cy, fx, fy;
    int depth_width, depth_heigth;

    double depth_maxdist, depth_mindist;
    int depth_filter_margin;
    double k_depth_scaling_factor;
    int skip_pixel;

    Eigen::Vector3d camera_pos;
    Eigen::Quaterniond camera_q;

    Eigen::Matrix3d R_C_2_W, R_C_2_B;
    Eigen::Vector3d T_C_2_B, T_C_2_W;

    // 点云位姿变换（直接从里程计获取，用于点云变换）
    Eigen::Matrix3d R_PC_2_W;  // 点云到世界坐标系的旋转矩阵
    Eigen::Vector3d T_PC_2_W;  // 点云到世界坐标系的平移向量
    Eigen::Quaterniond pointcloud_q;  // 点云的四元数

    cv::Mat depth_image;
    pcl::PointCloud<pcl::PointXYZ> ptws_hit, ptws_miss;
};

using PointType = pcl::PointXYZI;
boost::shared_ptr<PatchWorkpp<PointType>> ground_segmentation;
// 两个发布者，用于发布地面和非地面点云
ros::Publisher ground_pub;
ros::Publisher nonground_pub;

cameraData camData_;

SOGMMap map_;

std::string frame_id_;
std::string child_frame_id_;

// ROS定时器，用于定期触发某些操作，例如更新地图
ros::Timer map_update_timer_;
// ROS定时器，用于定期发布局部地图状态
ros::Timer map_vis_timer_;
std::mutex map_mutex_;        // <<-- 用于保护共享对象 map_ 的互斥锁

// 修改同步策略为两个话题的同步：点云和里程计
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> SyncPolicyPointCloudOdom;
typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyPointCloudOdom>> SynchronizerPointCloudOdom;
SynchronizerPointCloudOdom sync_pointcloud_odom_;
// 两个智能指针，分别指向点云和里程计数据的订阅者
std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> pointcloud_sub_;
std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;

// for visualization
// 发布滑动窗口大小
ros::Publisher slide_global_map_range_pub_, local_update_range_pub_; // the sliding window size

// 多分辨率占据和空闲区域
// ros::Publisher multi_res_occ_pub_, multi_res_free_pub_;
// 新增的Publisher，用于发布局部地图的完整状态
ros::Publisher local_map_state_pub_; 

// 可选的深度图发布器，用于调试
ros::Publisher generated_depth_pub_front_, generated_depth_pub_left_, generated_depth_pub_right_, generated_depth_pub_back_;

bool depth_need_update_;
Eigen::Vector3d local_map_boundary_min_, local_map_boundary_max_;

// visualization
// 局部更新范围的可视化标记
void publishLocalUpdateRange()
{
    Eigen::Vector3d map_min_pos, map_max_pos, cube_pos, cube_scale;
    visualization_msgs::Marker mk;

    // 局部地图边界的最小和最大位置
    map_min_pos = local_map_boundary_min_;
    map_max_pos = local_map_boundary_max_;

    // 计算立方体的中心位置和大小
    // cube_pos 被计算为 map_min_pos 和 map_max_pos 的中点
    // cube_scale 则是这两个位置的差值，表示立方体的尺寸
    cube_pos = 0.5 * (map_min_pos + map_max_pos);
    cube_scale = map_max_pos - map_min_pos;

    mk.header.frame_id = frame_id_;
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;
    mk.id = 0;

    mk.pose.position.x = cube_pos(0);
    mk.pose.position.y = cube_pos(1);
    mk.pose.position.z = cube_pos(2);

    mk.scale.x = cube_scale(0);
    mk.scale.y = cube_scale(1);
    mk.scale.z = cube_scale(2);

    mk.color.a = 0.2;
    mk.color.r = 1.0;
    mk.color.g = 1.0;
    mk.color.b = 0.0;

    mk.pose.orientation.w = 1.0;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    
    // mk是一个visualization_msgs::Marker类型的消息，表示一个局部范围的立方体
    // 下面的代码将这个消息发布到 local_update_range_pub_ 话题上
    local_update_range_pub_.publish(mk);
}

// 发布一个表示全局网格地图范围的可视化标记
void publishSlideGlobalGridMapRange()
{
    Eigen::Vector3d map_min_pos, map_max_pos, cube_pos, cube_scale;
    visualization_msgs::Marker mk;

    map_.blockIdxToWorld(map_.getOrigin(), map_min_pos);
    map_.blockIdxToWorld(map_.getOrigin() + map_.getNum3dim(), map_max_pos);
    // map_min_pos = local_map_boundary_min_;
    // map_max_pos = local_map_boundary_max_;

    cube_pos = 0.5 * (map_min_pos + map_max_pos);
    cube_scale = map_max_pos - map_min_pos;

    mk.header.frame_id = frame_id_;
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;
    mk.id = 0;

    mk.pose.position.x = cube_pos(0);
    mk.pose.position.y = cube_pos(1);
    mk.pose.position.z = cube_pos(2);

    mk.scale.x = cube_scale(0);
    mk.scale.y = cube_scale(1);
    mk.scale.z = cube_scale(2);

    mk.color.a = 0.2;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;

    mk.pose.orientation.w = 1.0;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;

    // 发布滑动窗口范围的可视化标记
    slide_global_map_range_pub_.publish(mk);
}

// 发布全局占据地图的点云数据
void publishSlideGlobalOccMap()
{
    pcl::PointXYZ pt;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    Eigen::Vector3d pos;
    int index;

    // 原点和滑动窗口距离原点最远的对角点
    Eigen::Vector3i min_cut, max_cut;
    min_cut = map_.getOrigin();
    max_cut = map_.getOrigin() + map_.getNum3dim();

    // 遍历整个滑动窗口范围内的体素
    // 将每个体素的体素坐标转换为索引，然后检查该体素是否被占据
    // 如果该体素被占据，则将索引转换为世界坐标，添加到点云数据中
    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = min_cut(2); z <= max_cut(2); ++z)
            {
                map_.blockIdxToLocalLinear(Eigen::Vector3i(x, y, z), index);
                if (!map_.isOccupied(index))
                    continue;

                Eigen::Vector3i blockIdx = map_.linearToBlockIdx(index);
                map_.blockIdxToWorld(blockIdx, pos);

                pt.x = pos(0);
                pt.y = pos(1);
                pt.z = pos(2);
                cloud.points.push_back(pt);
            }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = frame_id_;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
}

/**
 * @brief 将点云转换为深度图
 * @param cloud_pcl 输入的PCL点云
 * @param R_C_2_W 相机到世界坐标系的旋转矩阵
 * @param T_C_2_W 相机到世界坐标系的平移向量
 * @return 生成的深度图
 */
cv::Mat convertPointCloudToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_pcl)
{
    // 初始化深度图
    cv::Mat depth_image = cv::Mat::zeros(camData_.depth_heigth, camData_.depth_width, CV_32FC1);
    
    // 处理点云中的每个点
    for (const auto& point : cloud_pcl->points)
    {
        // 检查点是否有效（非NaN）
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;
        
        // 将点从激光雷达坐标系转换到相机坐标系
        Eigen::Vector3d pt_camera(-point.y, -point.z, point.x);
        
        // 过滤相机后方的点
        if (pt_camera.z() <= 0)
            continue;
        
        // 计算距离并过滤超出范围的点
        double distance = pt_camera.norm();
        if (distance < camData_.depth_mindist || distance > camData_.depth_maxdist)
            continue;
        
        // 使用针孔相机模型投影到像素坐标
        double u = camData_.fx * (pt_camera.x() / pt_camera.z()) + camData_.cx;
        double v = camData_.fy * (pt_camera.y() / pt_camera.z()) + camData_.cy;
        
        // 转换为整数像素坐标
        int u_int = static_cast<int>(std::round(u));
        int v_int = static_cast<int>(std::round(v));
        
        // 检查是否在图像范围内
        if (u_int >= 0 && u_int < camData_.depth_width && v_int >= 0 && v_int < camData_.depth_heigth)
        {
            // 获取当前像素的深度值
            float& current_depth = depth_image.at<float>(v_int, u_int);
            
            // 如果当前像素没有深度值，或者新的深度值更近，则更新
            if (current_depth == 0.0f || pt_camera.z() < current_depth)
            {
                current_depth = static_cast<float>(pt_camera.z());
            }
        }
    }
    
    // 转换为16位无符号整型（与原始深度图格式匹配）
    cv::Mat depth_image_16u;
    depth_image.convertTo(depth_image_16u, CV_16UC1, camData_.k_depth_scaling_factor);
    
    return depth_image_16u;
}

/**
 * @brief 点云和里程计数据的回调函数 (最终修正版：完整处理四虚拟相机及外参)
 * @param cloud_msg 完整的360度点云数据
 * @param odom 里程计数据
 */
void pointCloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, 
                           const nav_msgs::OdometryConstPtr &odom)
{
    static int update_num = 0;
    static double occ_all_t = 0;
    static double occ_max_t = 0;

    ros::Time t1, t2;
    t1 = ros::Time::now();

    // ======================== 1. 提取基础位姿信息 ========================
    camData_.camera_pos(0) = odom->pose.pose.position.x;
    camData_.camera_pos(1) = odom->pose.pose.position.y;
    camData_.camera_pos(2) = odom->pose.pose.position.z;

    Eigen::Quaterniond lidar_q(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                               odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    Eigen::Matrix3d R_LIDAR_2_W = lidar_q.toRotationMatrix();
    Eigen::Vector3d T_LIDAR_2_W = camData_.camera_pos;

    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(tf::Transform(
                                              tf::Quaternion(lidar_q.x(), lidar_q.y(), lidar_q.z(), lidar_q.w()),
                                              tf::Vector3(T_LIDAR_2_W(0), T_LIDAR_2_W(1), T_LIDAR_2_W(2))),
                                          odom->header.stamp, frame_id_, child_frame_id_));

    // ======================== 2. 点云预处理和分割 ========================
    // pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<PointType>::Ptr full_cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*cloud_msg, *full_cloud);

    pcl::PointCloud<PointType> ground_cloud;
    pcl::PointCloud<PointType> nonground_cloud;
    double time_taken;

    ground_segmentation->estimate_ground(*full_cloud, ground_cloud, nonground_cloud, time_taken);

    sensor_msgs::PointCloud2 ground_msg, nonground_msg;
    pcl::toROSMsg(ground_cloud, ground_msg);
    ground_msg.header = cloud_msg->header;

    pcl::toROSMsg(nonground_cloud, nonground_msg);
    nonground_msg.header = cloud_msg->header;

    ground_pub.publish(ground_msg);
    nonground_pub.publish(nonground_msg);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_clouds(4);
    for(int i=0; i<4; ++i) {
        segmented_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
    }

    const double rad_45 = M_PI / 4.0;
    const double rad_135 = 3.0 * M_PI / 4.0;

    for (const auto& point : full_cloud->points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        
        double angle = std::atan2(-point.y, point.x);

        if (std::abs(angle) <= rad_45) { // 前方
            segmented_clouds[0]->points.emplace_back(point.x, point.y, point.z);
        } else if (angle > rad_45 && angle <= rad_135) { // 左方
            segmented_clouds[1]->points.emplace_back(point.x, point.y, point.z);
        } else if (angle < -rad_45 && angle >= -rad_135) { // 右方
            segmented_clouds[2]->points.emplace_back(point.x, point.y, point.z);
        } else { // 后方
            segmented_clouds[3]->points.emplace_back(point.x, point.y, point.z);
        }
    }

    // ======================== 3. 循环处理四个虚拟相机 ========================
    
    std::vector<Eigen::Matrix3d> virtual_cam_rotations(4);
    virtual_cam_rotations[0] = Eigen::Matrix3d::Identity(); // 前
    virtual_cam_rotations[1] = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();  // 左
    virtual_cam_rotations[2] = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(); // 右
    virtual_cam_rotations[3] = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();        // 后
    
    // 定义四个方向的发布器数组
    std::vector<ros::Publisher*> depth_publishers = {
        &generated_depth_pub_front_,
        &generated_depth_pub_left_,
        &generated_depth_pub_right_,
        &generated_depth_pub_back_
    };
    
    std::vector<std::string> direction_names = {"front", "left", "right", "back"};

    std::lock_guard<std::mutex> lock(map_mutex_);

    for (int i = 0; i < 4; ++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud = segmented_clouds[i];
        if (current_cloud->empty()) continue;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_for_depth(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Matrix3d rot_to_front = virtual_cam_rotations[i].transpose();
        for (const auto& point : current_cloud->points)
        {
            Eigen::Vector3d p(point.x, point.y, point.z);
            Eigen::Vector3d rotated_p = rot_to_front * p;
            cloud_for_depth->points.push_back(pcl::PointXYZ(rotated_p.x(), rotated_p.y(), rotated_p.z()));
        }

        cv::Mat depth_image = convertPointCloudToDepthImage(cloud_for_depth);

        // ======================== 新增：发布当前方向的深度图 ========================
        if (depth_publishers[i]->getNumSubscribers() > 0)
        {
            try {
                sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(
                    cloud_msg->header, 
                    "16UC1",  // 16位无符号整型深度图
                    depth_image
                ).toImageMsg();
                
                // 修改frame_id以区分不同方向
                depth_msg->header.frame_id = frame_id_ + "_" + direction_names[i];
                depth_publishers[i]->publish(depth_msg);
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception for %s depth image: %s", direction_names[i].c_str(), e.what());
            }
        }

        pcl::PointCloud<pcl::PointXYZ> ptws_hit;
        for (const auto& point : current_cloud->points)
        {
            Eigen::Vector3d pt_c(point.x, point.y, point.z);
            if (pt_c.norm() >= camData_.depth_mindist && pt_c.norm() <= camData_.depth_maxdist)
            {
                Eigen::Vector3d pt_w = R_LIDAR_2_W * pt_c + T_LIDAR_2_W;
                ptws_hit.points.push_back(pcl::PointXYZ(pt_w(0), pt_w(1), pt_w(2)));
            }
        }
        if (ptws_hit.empty()) continue;

        // c. 计算当前虚拟相机的完整世界位姿 (包含R_C_2_B外参)
        Eigen::Matrix3d R_VIRTUAL_2_W = R_LIDAR_2_W * virtual_cam_rotations[i];
        
        // ======================== 修正1: 旋转部分 ========================
        // 最终的光学坐标系旋转 = (世界 <- 载体) * (载体 <- 虚拟相机) * (虚拟相机 <- 光学坐标系)
        // 这里的 camData_.R_C_2_B 就是 (虚拟相机 <- 光学坐标系) 的旋转
        Eigen::Matrix3d R_FINAL_CAM_2_W = R_VIRTUAL_2_W * camData_.R_C_2_B;

        // ======================== 修正2: 平移部分 ========================
        // 最终的光学坐标系平移 = 载体世界平移 + 载体世界旋转 * (虚拟相机相对载体平移 + 虚拟相机旋转 * 光学坐标系相对虚拟相机平移)
        // T_C_W = T_B_W + R_B_W * T_V_B + R_B_W * R_V_B * T_C_V
        // T_V_B 为零, T_C_V 就是 camData_.T_C_2_B
        Eigen::Vector3d T_FINAL_CAM_2_W = T_LIDAR_2_W + R_VIRTUAL_2_W * camData_.T_C_2_B;
        
        // d. 使用修正后的完整位姿调用地图更新函数
        pcl::PointCloud<pcl::PointXYZ> ptws_miss;
        map_.update(&ptws_hit, &ptws_miss, depth_image, R_FINAL_CAM_2_W, T_FINAL_CAM_2_W, camData_.camera_pos);
    }
    t2 = ros::Time::now();
    
    publishLocalUpdateRange();
    publishSlideGlobalGridMapRange();

    update_num++;
    occ_all_t = occ_all_t + (t2 - t1).toSec();
    if ((t2 - t1).toSec() > occ_max_t)
        occ_max_t = (t2 - t1).toSec();

    if (update_num % 10 == 0)
    {
        std::cout << "[Occupancy] "
                  << "max time: " << occ_max_t << ", average time: " << occ_all_t / update_num << ", time: " << (t2 - t1).toSec() << std::endl;
    }
}

// 低频回调：负责生成和发布完整的局部地图状态
void visualizeMapCallback(const ros::TimerEvent &)
{
    // 只有当有节点订阅时才执行这个耗时操作
    if (local_map_state_pub_.getNumSubscribers() == 0)
        return;

    multilayer::VoxelGridMsgArray state_msg;
    
    ros::Time t1, t2;
    t1 = ros::Time::now();
    {
        // 在访问共享的 map_ 对象前加锁
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        // 调用耗时的 getLocalMapState 来填充消息
        map_.getLocalMapState(state_msg.voxels);
    } // 锁在这里被释放，锁的范围应尽可能小
    t2 = ros::Time::now();

    // 如果没有获取到任何占据单元，则不发布
    if (state_msg.voxels.empty())
        return;

    // 填充消息头并发布
    state_msg.header.frame_id = frame_id_;
    state_msg.header.stamp = ros::Time::now();
    local_map_state_pub_.publish(state_msg);

    // ROS_INFO_STREAM("Visualize map done, time: " << (t2 - t1).toSec());
}


/**
 * @brief 从 YAML 文件中设置激光雷达参数并计算对应的深度图参数。
 *
 * 该函数从指定的 YAML 文件中读取激光雷达参数，并使用这些参数计算对应的深度图参数。
 * 它期望 YAML 文件包含一个名为 "LiDAR" 的节点，该节点具有以下字段：
 * - fov_theta_range: 水平视场角范围 [min, max] (度)
 * - fov_phi_range: 垂直视场角范围 [min, max] (度)
 * - fov_depth: 最大探测距离 (米)
 * - sensor_res_hor: 水平分辨率 (度)
 * - sensor_res_vert: 垂直分辨率 (度)
 * - depth_mindist: 最小探测距离 (米)
 * - depth_filter_margin: 深度过滤的边距
 * - skip_pixel: 处理过程中跳过的像素数
 * - k_depth_scaling_factor: 深度值的缩放因子
 * - R_C_2_B: 从相机坐标系到机体坐标系的旋转矩阵
 * - T_C_2_B: 从相机坐标系到机体坐标系的平移向量
 *
 * @param filename 包含激光雷达参数的 YAML 文件的路径。
 */
void setCameraParam(std::string filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "**ERROR CAN NOT OPEN YAML FILE**" << std::endl;
    }

    cv::FileNode yaml_node = fs["LiDAR"];
    
    // 读取激光雷达参数
    cv::FileNode fov_theta_node = yaml_node["fov_theta_range"];
    cv::FileNode fov_phi_node = yaml_node["fov_phi_range"];
    
    double fov_theta_min = (double)fov_theta_node[0];
    double fov_theta_max = (double)fov_theta_node[1];
    double fov_phi_min = (double)fov_phi_node[0];
    double fov_phi_max = (double)fov_phi_node[1];
    
    double fov_depth = (double)yaml_node["fov_depth"];
    double sensor_res_hor = (double)yaml_node["sensor_res_hor"];
    double sensor_res_vert = (double)yaml_node["sensor_res_vert"];
    
    // 计算深度图尺寸（参考point2depth.py的计算方法）
    camData_.depth_width = static_cast<int>((fov_theta_max - fov_theta_min) / sensor_res_hor) + 1;
    camData_.depth_heigth = static_cast<int>((fov_phi_max - fov_phi_min) / sensor_res_vert) + 1;
    
    // 计算相机内参（参考point2depth.py的计算方法）
    // 将视场角转换为弧度
    double fov_h_rad = (fov_theta_max - fov_theta_min) * M_PI / 180.0;
    double fov_v_rad = (fov_phi_max - fov_phi_min) * M_PI / 180.0;
    
    // 使用针孔相机模型计算焦距
    camData_.fx = camData_.depth_width / (2.0 * tan(fov_h_rad / 2.0));
    camData_.fy = camData_.depth_heigth / (2.0 * tan(fov_v_rad / 2.0));
    
    // 主点设置为图像中心
    camData_.cx = (camData_.depth_width - 1) / 2.0;
    camData_.cy = (camData_.depth_heigth - 1) / 2.0;

    // 读取其他参数
    camData_.k_depth_scaling_factor = (double)(yaml_node["k_depth_scaling_factor"]);
    camData_.depth_maxdist = fov_depth;  // 使用激光雷达的最大探测距离
    camData_.depth_mindist = (double)(yaml_node["depth_mindist"]);
    camData_.depth_filter_margin = (double)(yaml_node["depth_filter_margin"]);
    camData_.skip_pixel = (double)(yaml_node["skip_pixel"]);

    std::cout << "[LiDAR INIT] 从 YAML 文件中读取激光雷达参数" << std::endl;

    cv::Mat rc2b, tc2b;
    yaml_node["R_C_2_B"] >> rc2b;
    yaml_node["T_C_2_B"] >> tc2b;

    cv::cv2eigen(rc2b, camData_.R_C_2_B);
    cv::cv2eigen(tc2b, camData_.T_C_2_B);

    std::cout << "[LiDAR INIT] 激光雷达参数初始化" << std::endl;
    std::cout << "[LiDAR INIT] 水平视场角范围: [" << fov_theta_min << ", " << fov_theta_max << "] 度" << std::endl;
    std::cout << "[LiDAR INIT] 垂直视场角范围: [" << fov_phi_min << ", " << fov_phi_max << "] 度" << std::endl;
    std::cout << "[LiDAR INIT] 最大探测距离: " << fov_depth << " 米" << std::endl;
    std::cout << "[LiDAR INIT] 水平分辨率: " << sensor_res_hor << " 度" << std::endl;
    std::cout << "[LiDAR INIT] 垂直分辨率: " << sensor_res_vert << " 度" << std::endl;
    std::cout << "[LiDAR INIT] 计算的深度图尺寸: " << camData_.depth_width << " x " << camData_.depth_heigth << std::endl;
    std::cout << "[LiDAR INIT] 计算的相机内参:" << std::endl;
    std::cout << "[LiDAR INIT] fx: " << camData_.fx << ", fy: " << camData_.fy << std::endl;
    std::cout << "[LiDAR INIT] cx: " << camData_.cx << ", cy: " << camData_.cy << std::endl;
    std::cout << "[LiDAR INIT] depth_maxdist: " << camData_.depth_maxdist << std::endl;
    std::cout << "[LiDAR INIT] depth_mindist: " << camData_.depth_mindist << std::endl;
    std::cout << "[LiDAR INIT] depth_filter_margin: " << camData_.depth_filter_margin << std::endl;
    std::cout << "[LiDAR INIT] skip_pixel: " << camData_.skip_pixel << std::endl;
    std::cout << "[LiDAR INIT] k_depth_scaling_factor: " << camData_.k_depth_scaling_factor << std::endl;
    std::cout << "[LiDAR INIT] R_C_2_B: \n"
              << camData_.R_C_2_B << std::endl;
    std::cout << "[LiDAR INIT] T_C_2_B: " << camData_.T_C_2_B.transpose() << std::endl;
}

/**
 * @brief 主函数，初始化 ROS 节点并设置相关参数和回调函数。
 *
 * 该函数执行以下操作：
 * 1. 初始化 ROS 节点 "sogm_map"。
 * 2. 创建一个私有的 NodeHandle 对象，用于与 ROS 进行交互。
 * 3. 从参数服务器获取参数文件路径，并输出该路径。
 * 4. 调用 map_ 对象的 init 函数，使用参数文件进行初始化。
 * 5. 调用 setCameraParam 函数，从参数文件中设置相机参数。
 * 6. 创建一个 ROS 定时器，每隔 0.05 秒触发一次 updateMapCallback 回调函数。
 * 7. 创建深度图像和里程计数据的订阅者，并使用 ApproximateTime 同步策略进行同步。
 * 8. 注册深度图像和里程计数据的回调函数 depthOdomCallback。
 * 9. 创建发布者，用于发布局部更新范围、滑动窗口范围、新占据区域和新空闲区域的话题。
 * 10. 设置 depth_need_update_ 标志为 false。
 * 11. 调用 ros::spin() 进入 ROS 事件循环，等待回调函数的触发。
 *
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @return int 返回值，0 表示程序正常退出。
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "sogm_map");
    ros::NodeHandle node("~");

    ground_segmentation.reset(new PatchWorkpp<PointType>(&node));

    ground_pub = node.advertise<sensor_msgs::PointCloud2>("ground_points", 1);
    nonground_pub = node.advertise<sensor_msgs::PointCloud2>("nonground_points", 1);

    node.param<std::string>("frame_id", frame_id_, "map");
    node.param<std::string>("child_frame_id", child_frame_id_, "base_link");

    std::string filename;
    node.param<std::string>("paramfile/path", filename, "./src/gridmap/config/sogm_map.yaml");
    std::cout << "parameter file: " << filename << std::endl;

    std::cout << "[SOGM Map] map initialized00" << std::endl;
    map_.init(filename);
    std::cout << "[SOGM Map] map initialized" << std::endl;
    setCameraParam(filename);
    
    map_.setCameraParameters(
        camData_.fx, camData_.fy, camData_.cx, camData_.cy,
        camData_.depth_width, camData_.depth_heigth,
        camData_.k_depth_scaling_factor,
        camData_.depth_maxdist,
        camData_.depth_mindist,
        camData_.skip_pixel,
        camData_.R_C_2_B, camData_.T_C_2_B
    );

    // map_update_timer_ = node.createTimer(ros::Duration(0.05), updateMapCallback);
    map_vis_timer_ = node.createTimer(ros::Duration(1.0), visualizeMapCallback);

    // 修改订阅者为两个话题：点云和里程计
    pointcloud_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(node, "/pointcloud", 1));
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node, "/odom", 1));
    sync_pointcloud_odom_.reset(new message_filters::Synchronizer<SyncPolicyPointCloudOdom>(
        SyncPolicyPointCloudOdom(200), *pointcloud_sub_, *odom_sub_));
    sync_pointcloud_odom_->registerCallback(boost::bind(pointCloudOdomCallback, _1, _2));

    // 发布局部更新范围、滑动窗口范围、新占据区域和新空闲区域的话题
    local_update_range_pub_ = node.advertise<visualization_msgs::Marker>("/map/range/local", 10);
    slide_global_map_range_pub_ = node.advertise<visualization_msgs::Marker>("/map/range/slide", 10);

    // 添加新的发布器，用于发布局部地图状态
    local_map_state_pub_ = node.advertise<multilayer::VoxelGridMsgArray>("/map/local_state", 10);
    
    // 可选的深度图发布器，用于调试
    generated_depth_pub_front_ = node.advertise<sensor_msgs::Image>("/generated_depth/front", 10);
    generated_depth_pub_left_ = node.advertise<sensor_msgs::Image>("/generated_depth/left", 10);
    generated_depth_pub_right_ = node.advertise<sensor_msgs::Image>("/generated_depth/right", 10);
    generated_depth_pub_back_ = node.advertise<sensor_msgs::Image>("/generated_depth/back", 10);

    depth_need_update_ = false;

    ros::spin();

    return 0;
}