#include <iostream>
#include <memory>
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

#include <mutex>
#include <std_msgs/Bool.h>  // 添加这个头文件
#include <ctime>            // 添加这个头文件用于时间戳

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

// 这种同步策略允许在时间上近似匹配的消息对进行同步
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> SyncPolicyImageOdom;
typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;
SynchronizerImageOdom sync_image_odom_;
// 两个智能指针，分别指向深度图像和里程计数据的订阅者
std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;

// for visualization
// 发布滑动窗口大小
ros::Publisher slide_global_map_range_pub_, local_update_range_pub_; // the sliding window size

// 多分辨率占据和空闲区域
// ros::Publisher multi_res_occ_pub_, multi_res_free_pub_;
// 新增的Publisher，用于发布局部地图的完整状态
ros::Publisher local_map_state_pub_; 

bool depth_need_update_;
Eigen::Vector3d local_map_boundary_min_, local_map_boundary_max_;

// 地图保存
ros::Timer map_save_timer_;
bool enable_auto_save_ = false;
double save_interval_ = 60.0;  // 默认60秒保存一次
std::string save_directory_ = "./";

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

/**
 * @brief 深度图像和里程计数据的回调函数
 * @param img 深度图像
 * @param odom 里程计数据
 */
void depthOdomCallback(const sensor_msgs::ImageConstPtr &img, const nav_msgs::OdometryConstPtr &odom)
{
    // 从里程计数据中提取相机的位置和姿态
    camData_.camera_pos(0) = odom->pose.pose.position.x;
    camData_.camera_pos(1) = odom->pose.pose.position.y;
    camData_.camera_pos(2) = odom->pose.pose.position.z;

    // 这里的姿态表示载体坐标到世界坐标的变换
    // 以旋转矩阵表示为R_B_2_W，即R_{WB}
    camData_.camera_q = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
                                           odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);

    // TODO：没理解变换矩阵的含义
    // DONE！！！
    // pt_w(0) = (u - camData_.cx) * depth / camData_.fx;
    // pt_w(1) = (v - camData_.cy) * depth / camData_.fy;
    // pt_w(2) = depth;
    // pt_w = camData_.R_C_2_W * pt_w + camData_.T_C_2_W;
    // 从上述代码来看，R_C_2_W 是相机坐标到世界坐标的旋转矩阵，也就是R_{WC}
    // 下面公式即为：R_{WC}=R_{WB}*R_{BC}
    camData_.R_C_2_W = camData_.camera_q.toRotationMatrix() * camData_.R_C_2_B;
    // T_{WC}=T_{WB}+R_{WB}*T_{BC}，但是由于T_{BC}给定为0，所以这里没有*R_{WB}项，不影响
    camData_.T_C_2_W = camData_.camera_pos + camData_.T_C_2_B;

    // 发布相机的位姿信息
    // tf::StampedTransform 包含了一个变换（tf::Transform）
    // 一个时间戳（odom->header.stamp）
    // 以及两个坐标系的名称（"map" 和 "base_link"）
    static tf::TransformBroadcaster br;
    Eigen::Quaterniond eq(camData_.R_C_2_W);
    br.sendTransform(tf::StampedTransform(tf::Transform(
                                              tf::Quaternion(eq.w(), eq.x(), eq.y(), eq.z()),
                                              tf::Vector3(camData_.T_C_2_W(0), camData_.T_C_2_W(1), camData_.T_C_2_W(2))),
                                          odom->header.stamp, frame_id_, child_frame_id_));

    /* get depth image */
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
    // 应用深度缩放因子，将深度图像转换为16位无符号整型
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, camData_.k_depth_scaling_factor);
    }

    cv_ptr->image.copyTo(camData_.depth_image);

    camData_.ptws_hit.clear();
    camData_.ptws_miss.clear();

    Eigen::Vector3d pt_w;
    pcl::PointXYZ pt;
    double depth;

    uint16_t *row_ptr;
    int cols = camData_.depth_image.cols;
    int rows = camData_.depth_image.rows;

    const double inv_factor = 1.0 / camData_.k_depth_scaling_factor;

    if (true)
    {
        local_map_boundary_max_ = camData_.camera_pos;
        local_map_boundary_min_ = camData_.camera_pos;
    }

    // depth_filter_margin：参数为0，表示不对深度图像进行裁剪
    // skip_pixel：参数为4，表示每隔4个像素采样一次
    for (int v = camData_.depth_filter_margin; v < rows - camData_.depth_filter_margin; v += camData_.skip_pixel)
    {
        // ptr<uint16_t>(v) 方法用于获取深度图像中第 v 行的指针，并将其转换为指向 uint16_t 类型数据的指针
        // camData_.depth_filter_margin 是一个偏移量，以便跳过图像行的前几个像素
        row_ptr = camData_.depth_image.ptr<uint16_t>(v) + camData_.depth_filter_margin;

        for (int u = camData_.depth_filter_margin; u < cols - camData_.depth_filter_margin; u += camData_.skip_pixel)
        {
            depth = (*row_ptr) * inv_factor;
            row_ptr = row_ptr + camData_.skip_pixel;

            if (*row_ptr == 0 || depth > camData_.depth_maxdist)
            {
                depth = camData_.depth_maxdist;

                pt_w(0) = (u - camData_.cx) * depth / camData_.fx;
                pt_w(1) = (v - camData_.cy) * depth / camData_.fy;
                pt_w(2) = depth;
                // TODO！！理解变换矩阵
                // pt_{W}=R_{WC}*pt_{C}+T_{WC}
                // DONE！！！
                pt_w = camData_.R_C_2_W * pt_w + camData_.T_C_2_W;

                pt.x = pt_w(0);
                pt.y = pt_w(1);
                pt.z = pt_w(2);

                // 深度值超过最大深度值，则将深度值设置为最大深度距离
                // 将对应的三维点添加到未命中点云中
                camData_.ptws_miss.points.push_back(pt);
            }
            else if (depth < camData_.depth_mindist)
            {
                continue;
            }
            else
            {
                pt_w(0) = (u - camData_.cx) * depth / camData_.fx;
                pt_w(1) = (v - camData_.cy) * depth / camData_.fy;
                pt_w(2) = depth;
                pt_w = camData_.R_C_2_W * pt_w + camData_.T_C_2_W;

                pt.x = pt_w(0);
                pt.y = pt_w(1);
                pt.z = pt_w(2);

                camData_.ptws_hit.points.push_back(pt);
            }

            // 在每次添加点到点云后，代码更新局部地图的边界，确保边界包含所有处理过的点
            if (true)
            {
                local_map_boundary_max_(0) = std::max(local_map_boundary_max_(0), pt_w(0));
                local_map_boundary_max_(1) = std::max(local_map_boundary_max_(1), pt_w(1));
                local_map_boundary_max_(2) = std::max(local_map_boundary_max_(2), pt_w(2));

                local_map_boundary_min_(0) = std::min(local_map_boundary_min_(0), pt_w(0));
                local_map_boundary_min_(1) = std::min(local_map_boundary_min_(1), pt_w(1));
                local_map_boundary_min_(2) = std::min(local_map_boundary_min_(2), pt_w(2));
            }
        }
    }

    depth_need_update_ = true;
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
        map_.update(&camData_.ptws_hit, &camData_.ptws_miss, camData_.depth_image, camData_.R_C_2_W, camData_.T_C_2_W, camData_.camera_pos);
    }

    t2 = ros::Time::now();
    
    // 更新 local_map_boundary_min_ 和 local_map_boundary_max_
    // 将它们设置为相机位置
    local_map_boundary_min_ = camData_.camera_pos;
    local_map_boundary_max_ = camData_.camera_pos;

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

void saveSemanticKittiMap() {
    std::cout << "[SemanticKITTI] Auto-saving map at sensor position: " 
              << camData_.camera_pos.transpose() << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        bool success = map_.saveSemanticKittiVoxels(save_directory_, camData_.camera_pos);
        
        if (success) {
            std::cout << "[SemanticKITTI] Successfully auto-saved map to " << save_directory_ << std::endl;            
        } else {
            std::cerr << "[SemanticKITTI] Failed to auto-save map!" << std::endl;
        }
    }
}

// 定时器回调函数
void mapSaveTimerCallback(const ros::TimerEvent &) {
    if (!enable_auto_save_) {
        return;
    }
    
    // 检查地图是否有效（传感器位置不为零）
    if (camData_.camera_pos.norm() < 1e-6) {
        ROS_WARN("[SemanticKITTI] Skipping save: sensor position not initialized");
        return;
    }
    
    saveSemanticKittiMap();
}

/**
 * @brief 从 YAML 文件中设置相机参数。
 *
 * 该函数从指定的 YAML 文件中读取相机参数，并使用这些参数初始化相机数据结构。
 * 它期望 YAML 文件包含一个名为 "DepthCamera" 的节点，该节点具有以下字段：
 * - heigth: 深度相机图像的高度。
 * - width: 深度相机图像的宽度。
 * - fx: 深度相机在 x 方向上的焦距。
 * - fy: 深度相机在 y 方向上的焦距。
 * - cx: 深度相机在 x 方向上的主点。
 * - cy: 深度相机在 y 方向上的主点。
 * - k_depth_scaling_factor: 深度值的缩放因子。
 * - depth_maxdist: 深度相机可测量的最大距离。
 * - depth_mindist: 深度相机可测量的最小距离。
 * - depth_filter_margin: 深度过滤的边距。
 * - skip_pixel: 处理过程中跳过的像素数。
 * - R_C_2_B: 从相机坐标系到机体坐标系的旋转矩阵。
 * - T_C_2_B: 从相机坐标系到机体坐标系的平移向量。
 *
 * @param filename 包含相机参数的 YAML 文件的路径。
 */
void setCameraParam(std::string filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "**ERROR CAN NOT OPEN YAML FILE**" << std::endl;
    }

    cv::FileNode yaml_node = fs["DepthCamera"];
    camData_.depth_heigth = (int)(yaml_node["height"]);
    camData_.depth_width = (int)(yaml_node["width"]);
    camData_.fx = (double)(yaml_node["fx"]);
    camData_.fy = (double)(yaml_node["fy"]);
    camData_.cx = (double)(yaml_node["cx"]);
    camData_.cy = (double)(yaml_node["cy"]);

    camData_.k_depth_scaling_factor = (double)(yaml_node["k_depth_scaling_factor"]);
    camData_.depth_maxdist = (double)(yaml_node["depth_maxdist"]);
    camData_.depth_mindist = (double)(yaml_node["depth_mindist"]);
    camData_.depth_filter_margin = (double)(yaml_node["depth_filter_margin"]);
    camData_.skip_pixel = (double)(yaml_node["skip_pixel"]);

    cv::Mat rc2b, tc2b;
    yaml_node["R_C_2_B"] >> rc2b;
    yaml_node["T_C_2_B"] >> tc2b;

    cv::cv2eigen(rc2b, camData_.R_C_2_B);
    cv::cv2eigen(tc2b, camData_.T_C_2_B);

    cv::FileNode mapsave_node = fs["SemanticKITTI"];
    int save_map = (int)(mapsave_node["enable_auto_save"]);
    enable_auto_save_ = (bool)(save_map);
    save_interval_ = (double)(mapsave_node["save_interval"]);
    save_directory_ = (std::string)(mapsave_node["save_directory"]);
    // 确保保存目录以 '/' 结尾
    if (!save_directory_.empty() && save_directory_.back() != '/') {
        save_directory_ += '/';
    }

    std::cout << "[CameraParam INIT] save map enable: " << (enable_auto_save_ ? "true" : "false") << std::endl;
    std::cout << "[CameraParam INIT] save interval: " << save_interval_ << " seconds" << std::endl;
    std::cout << "[CameraParam INIT] save directory: " << save_directory_ << std::endl;

    std::cout << "[CameraParam INIT] use depth camera" << std::endl;
    std::cout << "[CameraParam INIT] depth heigth: " << camData_.depth_heigth << std::endl;
    std::cout << "[CameraParam INIT] depth width: " << camData_.depth_width << std::endl;
    std::cout << "[CameraParam INIT] depth fx: " << camData_.fx << std::endl;
    std::cout << "[CameraParam INIT] depth fy: " << camData_.fy << std::endl;
    std::cout << "[CameraParam INIT] depth cx: " << camData_.cx << std::endl;
    std::cout << "[CameraParam INIT] depth cy: " << camData_.cy << std::endl;
    std::cout << "[CameraParam INIT] depth k_depth_scaling_factor: " << camData_.k_depth_scaling_factor << std::endl;
    std::cout << "[CameraParam INIT] depth depth_maxdist: " << camData_.depth_maxdist << std::endl;
    std::cout << "[CameraParam INIT] depth depth_mindist: " << camData_.depth_mindist << std::endl;
    std::cout << "[CameraParam INIT] depth depth_filter_margin: " << camData_.depth_filter_margin << std::endl;
    std::cout << "[CameraParam INIT] depth skip_pixel: " << camData_.skip_pixel << std::endl;
    std::cout << "[CameraParam INIT] R_C_2_B: \n"
              << camData_.R_C_2_B << std::endl;
    std::cout << "[CameraParam INIT] T_C_2_B: " << camData_.T_C_2_B.transpose() << std::endl;
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

    map_.init(filename);
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

    map_update_timer_ = node.createTimer(ros::Duration(0.05), updateMapCallback);
    map_vis_timer_ = node.createTimer(ros::Duration(1.0), visualizeMapCallback);

    // 新增：地图保存定时器（只有启用自动保存时才创建）
    if (enable_auto_save_ && save_interval_ > 0) {
        map_save_timer_ = node.createTimer(ros::Duration(save_interval_), mapSaveTimerCallback);
        ROS_INFO("[SemanticKITTI] Auto-save timer created with interval: %.1f seconds", save_interval_);
    }

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node, "/depth", 1));
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node, "/odom", 1));
    sync_image_odom_.reset(new message_filters::Synchronizer<SyncPolicyImageOdom>(SyncPolicyImageOdom(100), *depth_sub_, *odom_sub_));
    sync_image_odom_->registerCallback(boost::bind(depthOdomCallback, _1, _2));

    // 发布局部更新范围、滑动窗口范围、新占据区域和新空闲区域的话题
    local_update_range_pub_ = node.advertise<visualization_msgs::Marker>("/map/range/local", 10);
    slide_global_map_range_pub_ = node.advertise<visualization_msgs::Marker>("/map/range/slide", 10);
    local_map_state_pub_ = node.advertise<multilayer::VoxelGridMsgArray>("/map/local_state", 10);

    depth_need_update_ = false;

    ros::spin();

    return 0;
}