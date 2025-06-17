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

#include <multilayer/VoxelGridMsg.h>
#include <multilayer/VoxelGridMsgArray.h>

struct cameraData
{
    /* Spherical Projection Parameters (replaces pinhole camera model) */
    int polar_width, polar_height;
    double fov_theta_min_rad, fov_theta_max_rad;
    double fov_phi_min_rad, fov_phi_max_rad;
    double sensor_res_hor_rad, sensor_res_vert_rad;

    /* Common Parameters */
    double depth_maxdist, depth_mindist;
    double k_depth_scaling_factor;

    /* Pose Information */
    Eigen::Matrix3d R_WL;
    Eigen::Vector3d T_WL;
    Eigen::Quaterniond pointcloud_q;

    /* Data Buffers */
    cv::Mat depth_image;
    pcl::PointCloud<pcl::PointXYZ> ptws_hit, ptws_miss;
};

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
ros::Publisher generated_depth_pub_;

bool depth_need_update_;
Eigen::Vector3d local_map_boundary_min_, local_map_boundary_max_;

int Rad2Idx(double rad, bool is_horizontal) {
    if (is_horizontal) { // 水平方向 (theta)
        if (rad < camData_.fov_theta_min_rad || rad > camData_.fov_theta_max_rad) return -1;
        return static_cast<int>((rad - camData_.fov_theta_min_rad) / camData_.sensor_res_hor_rad);
    } else { // 垂直方向 (phi)
        if (rad < camData_.fov_phi_min_rad || rad > camData_.fov_phi_max_rad) return -1;
        return static_cast<int>((rad - camData_.fov_phi_min_rad) / camData_.sensor_res_vert_rad);
    }
}

// 将传感器坐标系下的笛卡尔坐标转换为球面坐标和深度图索引
void euc2polar(const Eigen::Vector3d& euc_pt, double& r, int& u, int& v) {
    r = euc_pt.norm();
    if (r < 1e-5) {
        u = -1; v = -1;
        return;
    }
    // D-Map 标准投影模型: X-前, Y-左, Z-上
    double theta_rad = atan2(euc_pt.y(), euc_pt.x()); // 方位角
    double phi_rad = atan2(euc_pt.z(), euc_pt.head<2>().norm()); // 俯仰角
    
    // 转换为像素索引 (u-col-width, v-row-height)
    u = Rad2Idx(theta_rad, true);
    v = Rad2Idx(phi_rad, false);
}

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
 * @return 生成的深度图
 */
cv::Mat convertPointCloudToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_pcl)
{
    // 初始化深度图 (使用浮点数以存储精确距离，最后再转换)
    cv::Mat depth_image = cv::Mat::zeros(camData_.polar_height, camData_.polar_width, CV_32FC1);
    
    // 处理点云中的每个点
    for (const auto& point : cloud_pcl->points)
    {
        double r; // 距离
        int u, v; // 像素坐标 (u-col-width, v-row-height)

        Eigen::Vector3d pt_sensor_frame(point.x, point.y, point.z);
        
        // 执行球面投影
        euc2polar(pt_sensor_frame, r, u, v);
        
        // 检查距离和投影坐标是否有效
        if (r < camData_.depth_mindist || r > camData_.depth_maxdist) continue;
        if (u < 0 || u >= camData_.polar_width || v < 0 || v >= camData_.polar_height) continue;
        
        // Z-buffering: 只保留最近的点
        float& current_depth = depth_image.at<float>(v, u);
        if (current_depth == 0.0f || r < current_depth) {
            current_depth = static_cast<float>(r);
        }
    }
    
    return depth_image;
}

/**
 * @brief 点云和里程计数据的回调函数 (最终修正版：完整处理四虚拟相机及外参)
 * @param cloud_msg 完整的360度点云数据
 * @param odom 里程计数据
 */
void pointCloudOdomCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg, 
                           const nav_msgs::OdometryConstPtr &odom)
{
<<<<<<< Updated upstream
    // 点云位姿变换
    camData_.pointcloud_q = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                              odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    camData_.R_WL = camData_.pointcloud_q.toRotationMatrix();
    camData_.T_WL = Eigen::Vector3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
=======
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
>>>>>>> Stashed changes

    static tf::TransformBroadcaster br;
    br.sendTransform(tf::StampedTransform(tf::Transform(
<<<<<<< Updated upstream
                                              tf::Quaternion(camData_.pointcloud_q.x(), camData_.pointcloud_q.y(),
                                                             camData_.pointcloud_q.z(), camData_.pointcloud_q.w()),
                                              tf::Vector3(camData_.T_WL(0), camData_.T_WL(1), camData_.T_WL(2))),
=======
                                              tf::Quaternion(lidar_q.x(), lidar_q.y(), lidar_q.z(), lidar_q.w()),
                                              tf::Vector3(T_LIDAR_2_W(0), T_LIDAR_2_W(1), T_LIDAR_2_W(2))),
>>>>>>> Stashed changes
                                          odom->header.stamp, frame_id_, child_frame_id_));

    // ======================== 2. 点云预处理和分割 ========================
    pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *full_cloud);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_clouds(4);
    for(int i=0; i<4; ++i) {
        segmented_clouds[i].reset(new pcl::PointCloud<pcl::PointXYZ>());
    }

<<<<<<< Updated upstream
    // 从点云生成深度图
    camData_.depth_image = convertPointCloudToDepthImage(cloud_pcl);
=======
    const double rad_45 = M_PI / 4.0;
    const double rad_135 = 3.0 * M_PI / 4.0;
>>>>>>> Stashed changes

    for (const auto& point : full_cloud->points)
    {
<<<<<<< Updated upstream
        sensor_msgs::Image depth_msg;
        cv_bridge::CvImage cv_bridge_image;
        cv_bridge_image.header.stamp = odom->header.stamp;
        cv_bridge_image.header.frame_id = frame_id_;
        cv_bridge_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        cv_bridge_image.image = camData_.depth_image;
        cv_bridge_image.toImageMsg(depth_msg);
        generated_depth_pub_.publish(depth_msg);
    }

    // 初始化局部地图边界
    if (true)
    {
        local_map_boundary_max_ = camData_.T_WL;
        local_map_boundary_min_ = camData_.T_WL;
    }
=======
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        
        double angle = std::atan2(-point.y, point.x);

        if (std::abs(angle) <= rad_45) { // 前方
            segmented_clouds[0]->points.push_back(point);
        } else if (angle > rad_45 && angle <= rad_135) { // 左方
            segmented_clouds[1]->points.push_back(point);
        } else if (angle < -rad_45 && angle >= -rad_135) { // 右方
            segmented_clouds[2]->points.push_back(point);
        } else { // 后方
            segmented_clouds[3]->points.push_back(point);
        }
    }

    // ======================== 3. 循环处理四个虚拟相机 ========================
    
    std::vector<Eigen::Matrix3d> virtual_cam_rotations(4);
    virtual_cam_rotations[0] = Eigen::Matrix3d::Identity(); // 前
    virtual_cam_rotations[1] = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();  // 左
    virtual_cam_rotations[2] = Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix(); // 右
    virtual_cam_rotations[3] = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();        // 后
    
    std::lock_guard<std::mutex> lock(map_mutex_);
>>>>>>> Stashed changes

    for (int i = 0; i < 4; ++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud = segmented_clouds[i];
        if (current_cloud->empty()) continue;

<<<<<<< Updated upstream
        Eigen::Vector3d pt_sensor_frame(point.x, point.y, point.z);
        double distance = pt_sensor_frame.norm();

        // 只处理距离在有效范围内的点
        if (distance >= camData_.depth_mindist && distance <= camData_.depth_maxdist)
        {
            // 使用点云专用变换到世界坐标系
            Eigen::Vector3d pt_w = camData_.R_WL * pt_sensor_frame + camData_.T_WL;

            camData_.ptws_hit.points.emplace_back(pt_w(0), pt_w(1), pt_w(2));
=======
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_for_depth(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Matrix3d rot_to_front = virtual_cam_rotations[i].transpose();
        for (const auto& point : current_cloud->points)
        {
            Eigen::Vector3d p(point.x, point.y, point.z);
            Eigen::Vector3d rotated_p = rot_to_front * p;
            cloud_for_depth->points.push_back(pcl::PointXYZ(rotated_p.x(), rotated_p.y(), rotated_p.z()));
        }

        cv::Mat depth_image = convertPointCloudToDepthImage(cloud_for_depth);
>>>>>>> Stashed changes

        pcl::PointCloud<pcl::PointXYZ> ptws_hit;
        for (const auto& point : current_cloud->points)
        {
            Eigen::Vector3d pt_c(point.x, point.y, point.z);
            if (pt_c.norm() >= camData_.depth_mindist && pt_c.norm() <= camData_.depth_maxdist)
            {
<<<<<<< Updated upstream
                local_map_boundary_max_.x() = std::max(local_map_boundary_max_.x(), pt_w.x());
                local_map_boundary_max_.y() = std::max(local_map_boundary_max_.y(), pt_w.y());
                local_map_boundary_max_.z() = std::max(local_map_boundary_max_.z(), pt_w.z());
                local_map_boundary_min_.x() = std::min(local_map_boundary_min_.x(), pt_w.x());
                local_map_boundary_min_.y() = std::min(local_map_boundary_min_.y(), pt_w.y());
                local_map_boundary_min_.z() = std::min(local_map_boundary_min_.z(), pt_w.z());
            }
        }
        else if (distance > camData_.depth_maxdist)
        {
            // 如果点在最大距离之外，添加到未命中的点云
            Eigen::Vector3d pt_w = camData_.R_WL * pt_sensor_frame + camData_.T_WL;
            camData_.ptws_miss.points.emplace_back(pt_w(0), pt_w(1), pt_w(2));
        }
=======
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
>>>>>>> Stashed changes
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

// 回调函数，用于更新地图数据并发布相关的可视化信息
// 该函数通过 ROS 定时器事件触发
void updateMapCallback(const ros::TimerEvent &)
{
    // update_num 用于记录更新次数
    // occ_all_t 和 occ_max_t 分别用于记录总的占据时间和最大占据时间
    static int update_num = 0;
    static double occ_all_t = 0;
    static double occ_max_t = 0;

    if (depth_need_update_ != true)
        return;

    ros::Time t1, t2, t3, t4;
    t1 = ros::Time::now();
    {
        // 在访问共享的 map_ 对象前加锁
        std::lock_guard<std::mutex> lock(map_mutex_);
        map_.update(&camData_.ptws_hit, &camData_.ptws_miss, camData_.depth_image, camData_.R_WL, camData_.T_WL);
    }

    t2 = ros::Time::now();
    
    // 更新 local_map_boundary_min_ 和 local_map_boundary_max_
    // 将它们设置为相机位置
    local_map_boundary_min_ = camData_.T_WL;
    local_map_boundary_max_ = camData_.T_WL;

    publishLocalUpdateRange(); // 发布局部更新范围
    publishSlideGlobalGridMapRange(); // 发布全局网格地图范围

    depth_need_update_ = false;

    update_num++;
    occ_all_t = occ_all_t + (t2 - t1).toSec();
    if ((t2 - t1).toSec() > occ_max_t)
        occ_max_t = (t2 - t1).toSec();

    if (update_num % 10 == 0)
    {
        std::cout << "[Occupancy] "
                  << "max time: " << occ_max_t << ", average time: " << occ_all_t / update_num << ", time: " << (t2 - t1).toSec() << std::endl;
    }
    // ROS_INFO_STREAM("update map done, update num: " << update_num);
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
    if (!fs.isOpened()) {
        std::cerr << "**ERROR CAN NOT OPEN YAML FILE**" << std::endl;
        return;
    }

    cv::FileNode yaml_node = fs["LiDAR"];
    
    cv::FileNode fov_theta_node = yaml_node["fov_theta_range"];
    cv::FileNode fov_phi_node = yaml_node["fov_phi_range"];
    
    double fov_theta_min_deg = (double)fov_theta_node[0];
    double fov_theta_max_deg = (double)fov_theta_node[1];
    double fov_phi_min_deg = (double)fov_phi_node[0];
    double fov_phi_max_deg = (double)fov_phi_node[1];
    
    // 转换为弧度
    camData_.fov_theta_min_rad = fov_theta_min_deg * M_PI / 180.0;
    camData_.fov_theta_max_rad = fov_theta_max_deg * M_PI / 180.0;
    camData_.fov_phi_min_rad = fov_phi_min_deg * M_PI / 180.0;
    camData_.fov_phi_max_rad = fov_phi_max_deg * M_PI / 180.0;
    
    double sensor_res_hor_deg = (double)yaml_node["sensor_res_hor"];
    double sensor_res_vert_deg = (double)yaml_node["sensor_res_vert"];
    camData_.sensor_res_hor_rad = sensor_res_hor_deg * M_PI / 180.0;
    camData_.sensor_res_vert_rad = sensor_res_vert_deg * M_PI / 180.0;

    // 计算深度图尺寸
    camData_.polar_width = static_cast<int>(std::ceil((camData_.fov_theta_max_rad - camData_.fov_theta_min_rad) / camData_.sensor_res_hor_rad));
    camData_.polar_height = static_cast<int>(std::ceil((camData_.fov_phi_max_rad - camData_.fov_phi_min_rad) / camData_.sensor_res_vert_rad));

    // 读取其他通用参数
    camData_.k_depth_scaling_factor = (double)(yaml_node["k_depth_scaling_factor"]);
    camData_.depth_maxdist = (double)yaml_node["fov_depth"];
    camData_.depth_mindist = (double)(yaml_node["depth_mindist"]);

    std::cout << "[LiDAR INIT] 使用球面投影模型初始化参数" << std::endl;
    std::cout << "[LiDAR INIT] 水平视场角范围: [" << fov_theta_min_deg << ", " << fov_theta_max_deg << "] 度" << std::endl;
    std::cout << "[LiDAR INIT] 垂直视场角范围: [" << fov_phi_min_deg << ", " << fov_phi_max_deg << "] 度" << std::endl;
    std::cout << "[LiDAR INIT] 计算的深度图尺寸 (宽x高): " << camData_.polar_width << " x " << camData_.polar_height << std::endl;
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
        camData_.fov_theta_min_rad, camData_.fov_theta_max_rad,
        camData_.fov_phi_min_rad, camData_.fov_phi_max_rad,
        camData_.sensor_res_hor_rad, camData_.sensor_res_vert_rad,
        camData_.polar_width, camData_.polar_height,
        camData_.k_depth_scaling_factor,
        camData_.depth_maxdist,
        camData_.depth_mindist,
        camData_.R_WL, camData_.T_WL
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
    generated_depth_pub_ = node.advertise<sensor_msgs::Image>("/generated_depth", 10);

    depth_need_update_ = false;

    ros::spin();

    return 0;
}