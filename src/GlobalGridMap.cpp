/*
 * @Name:
 * @Author:       yong
 * @Date: 2023-03-12 22:03:46
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2023-09-11 19:58:43
 * @Description:
 * @Subscriber:
 * @Publisher:
 */

#include "GlobalGridMap.hpp"

// 通过获取当前时间生成文件名，
// 并将全局网格地图中的点云数据保存为 PLY 文件，
// 从而实现了保存地图的功能。
bool GlobalGridMap::saveMap()
{
    time_t currentTime = time(NULL);
    char chCurrentTime[256];
    strftime(chCurrentTime, sizeof(chCurrentTime), "%Y%m%d %H%M%S", localtime(&currentTime));
    std::string stCurrentTime = chCurrentTime;
    std::string filename = stCurrentTime + "GridMap" + ".ply";

    pcl::PointCloud<pcl::PointXYZRGB> block_cloud, voxel_cloud, subvoxel_cloud;
    getMultiResMapCloud(block_cloud, voxel_cloud, subvoxel_cloud);
    pcl::io::savePLYFile(filename, block_cloud);
    pcl::io::savePLYFile(filename, voxel_cloud);
    pcl::io::savePLYFile(filename, subvoxel_cloud);

    return true;
}

void GlobalGridMap::init(ros::NodeHandle &nh)
{
    // 从参数服务器读取frame_id，如果没有设置则使用默认值
    nh.param<std::string>("frame_id", frame_id_, "map");

    save_map_sub_ = nh.subscribe("/map/save", 1, &GlobalGridMap::saveMapCallback, this);

    // 配置同步订阅
    multi_res_occ_sub_sync_.reset(new message_filters::Subscriber<multilayer::VoxelGridMsgArray>(nh, "/map/multi_res_occ", 10));
    multi_res_free_sub_sync_.reset(new message_filters::Subscriber<multilayer::VoxelGridMsgArray>(nh, "/map/multi_res_free", 10));
    
    // 创建同步器，设置队列大小为10
    sync_voxel_grids_.reset(new message_filters::Synchronizer<SyncPolicyVoxelGrids>(
        SyncPolicyVoxelGrids(10), *multi_res_occ_sub_sync_, *multi_res_free_sub_sync_));
    
    // 注册同步回调
    sync_voxel_grids_->registerCallback(boost::bind(&GlobalGridMap::syncVoxelGridsCallback, this, _1, _2));

    // 多分辨率
    // multi_res_occ_sub_ = nh.subscribe("/map/multi_res_occ", 10, &GlobalGridMap::multiResOccCallback, this);
    // multi_res_free_sub_ = nh.subscribe("/map/multi_res_free", 10, &GlobalGridMap::multiResFreeCallback, this);
    multi_res_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/multi_res_global", 10);

    // 添加分层级发布器
    block_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/block", 10);
    voxel_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/voxel", 10);
    subvoxel_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/subvoxel", 10);

    vis_timer_ = nh.createTimer(ros::Duration(0.1), &GlobalGridMap::visCallback, this);

    // 清空数据结构
    GLblocks_.clear();
    GLvoxels_.clear();
    GLsubvoxels_.clear();

    std::cout << "[GLOBAL MAP] init" << std::endl;
}

// 同步回调函数实现
void GlobalGridMap::syncVoxelGridsCallback(const multilayer::VoxelGridMsgArrayConstPtr& occ_msg,
                                           const multilayer::VoxelGridMsgArrayConstPtr& free_msg)
{
    // 处理占据体素
    std::vector<MultiResVoxel> occ_blocks;
    std::vector<MultiResVoxel> occ_voxels;
    std::vector<MultiResVoxel> occ_subvoxels;
    occ_blocks.reserve(occ_msg->voxels.size());
    occ_voxels.reserve(occ_msg->voxels.size());
    occ_subvoxels.reserve(occ_msg->voxels.size());
    
    // 创建占据体素查找集合，用于快速判断是否存在相同体素
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> occ_block_set;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> occ_voxel_set;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> occ_subvoxel_set;
    
    // 处理所有占据体素，同时构建查找集合
    for (const auto& voxel_msg : occ_msg->voxels) {
        MultiResVoxel voxel;
        voxel.center.x = voxel_msg.position.x;
        voxel.center.y = voxel_msg.position.y;
        voxel.center.z = voxel_msg.position.z;
        
        // 添加到查找集合
        // occ_voxel_set.insert(voxel);
        
        // 根据层级分类
        switch (voxel_msg.layer) {
            case 0: // BLOCK
                occ_blocks.push_back(voxel);
                occ_block_set.insert(voxel);
                break;
            case 1: // VOXEL
                occ_voxels.push_back(voxel);
                occ_voxel_set.insert(voxel);
                // std::cout << "Occ voxel: " << voxel.center.x << ", " << voxel.center.y << ", " << voxel.center.z << std::endl;
                break;
            case 2: // SUBVOXEL
                occ_subvoxels.push_back(voxel);
                occ_subvoxel_set.insert(voxel);
                break;
            default:
                break;
        }
    }
    
    // 处理空闲体素
    std::vector<MultiResVoxel> free_blocks;
    std::vector<MultiResVoxel> free_voxels;
    std::vector<MultiResVoxel> free_subvoxels;
    free_blocks.reserve(free_msg->voxels.size());
    free_voxels.reserve(free_msg->voxels.size());
    free_subvoxels.reserve(free_msg->voxels.size());
    
    // 处理所有空闲体素，但排除与占据体素相同的体素
    for (const auto& voxel_msg : free_msg->voxels) {
        MultiResVoxel voxel;
        voxel.center.x = voxel_msg.position.x;
        voxel.center.y = voxel_msg.position.y;
        voxel.center.z = voxel_msg.position.z;
        // voxel.layer = voxel_msg.layer;
        
        // 如果在占据体素集合中找到相同的体素，则跳过
        // if (occ_voxel_set.find(voxel) != occ_voxel_set.end()) {
        //     continue;
        // }
        
        // 根据层级分类
        switch (voxel_msg.layer) {
            case 0: // BLOCK
                if (occ_block_set.find(voxel) != occ_block_set.end()) {
                    continue;
                }
                free_blocks.push_back(voxel);
                break;
            case 1: // VOXEL
                // std::cout << "Free voxel: " << voxel.center.x << ", " << voxel.center.y << ", " << voxel.center.z << std::endl;
                if (occ_voxel_set.find(voxel) != occ_voxel_set.end()) {
                    continue;
                }
                free_voxels.push_back(voxel);
                break;
            case 2: // SUBVOXEL
                if (occ_subvoxel_set.find(voxel) != occ_subvoxel_set.end()) {
                    continue;
                }
                free_subvoxels.push_back(voxel);
                break;
            default:
                break;
        }
    }
    
    // 先添加占据体素，再移除空闲体素
    addMultiResVoxels(occ_blocks, occ_voxels, occ_subvoxels);
    removeMultiResVoxels(free_blocks, free_voxels, free_subvoxels);
    
    // std::cout << "[GLOBAL MAP] Synchronized update - Occ: " << occ_msg->voxels.size() 
    //           << ", Free: " << free_msg->voxels.size() << std::endl;
}

// 添加多个多分辨率体素
void GlobalGridMap::addMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                                      const std::vector<MultiResVoxel>& voxels,
                                      const std::vector<MultiResVoxel>& subvoxels) {
    for (const auto& block : blocks) {
        GLblocks_.insert(block);
    }
    for (const auto& voxel : voxels) {
        GLvoxels_.insert(voxel);
    }
    for (const auto& subvoxel : subvoxels) {
        GLsubvoxels_.insert(subvoxel);
        GLvoxels_.erase(subvoxel);
    }
}

// 移除多个多分辨率体素
void GlobalGridMap::removeMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                                          const std::vector<MultiResVoxel>& voxels,
                                          const std::vector<MultiResVoxel>& subvoxels) {
    for (const auto& block : blocks) {
        GLblocks_.erase(block);
    }
    for (const auto& voxel : voxels) {
        GLvoxels_.erase(voxel);
    }
    for (const auto& subvoxel : subvoxels) {
        GLsubvoxels_.erase(subvoxel);
    }
}

// 获取多分辨率地图点云（带颜色）
void GlobalGridMap::getMultiResMapCloud(pcl::PointCloud<pcl::PointXYZRGB> &block_cloud,
                                         pcl::PointCloud<pcl::PointXYZRGB> &voxel_cloud,
                                         pcl::PointCloud<pcl::PointXYZRGB> &subvoxel_cloud) {
    for (const auto& block : GLblocks_) {
        pcl::PointXYZRGB pt;
        pt.x = block.center.x;
        pt.y = block.center.y;
        pt.z = block.center.z;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        block_cloud.points.push_back(pt);
    }
    for (const auto& voxel : GLvoxels_) {
        pcl::PointXYZRGB pt;
        pt.x = voxel.center.x;
        pt.y = voxel.center.y;
        pt.z = voxel.center.z;
        pt.r = 0;
        pt.g = 255;
        pt.b = 0;
        voxel_cloud.points.push_back(pt);
    }
    for (const auto& subvoxel : GLsubvoxels_) {
        pcl::PointXYZRGB pt;
        pt.x = subvoxel.center.x;
        pt.y = subvoxel.center.y;
        pt.z = subvoxel.center.z;
        pt.r = 0;
        pt.g = 0;
        pt.b = 255;
        subvoxel_cloud.points.push_back(pt);
    }
    block_cloud.width = block_cloud.points.size();
    block_cloud.height = 1;
    block_cloud.is_dense = true;
    voxel_cloud.width = voxel_cloud.points.size();
    voxel_cloud.height = 1;
    voxel_cloud.is_dense = true;
    subvoxel_cloud.width = subvoxel_cloud.points.size();
    subvoxel_cloud.height = 1;
    subvoxel_cloud.is_dense = true;
}

// 发布多分辨率地图
void GlobalGridMap::publishMultiResMap() {
    pcl::PointCloud<pcl::PointXYZRGB> block_cloud, voxel_cloud, subvoxel_cloud;
    getMultiResMapCloud(block_cloud, voxel_cloud, subvoxel_cloud);
    block_cloud.header.frame_id = frame_id_;
    voxel_cloud.header.frame_id = frame_id_;
    subvoxel_cloud.header.frame_id = frame_id_;
    sensor_msgs::PointCloud2 block_msg, voxel_msg, subvoxel_msg;
    pcl::toROSMsg(block_cloud, block_msg);
    pcl::toROSMsg(voxel_cloud, voxel_msg);
    pcl::toROSMsg(subvoxel_cloud, subvoxel_msg);
    block_layer_pub_.publish(block_msg);
    voxel_layer_pub_.publish(voxel_msg);
    subvoxel_layer_pub_.publish(subvoxel_msg);
}

// 接收到保存地图的消息后，调用 saveMap() 函数，
// 将全局网格地图中的点云数据保存为 PLY 文件。
void GlobalGridMap::saveMapCallback(const std_msgs::BoolConstPtr &msg)
{
    if (msg->data == true)
    {
        saveMap();
    }
}

// 定时器回调函数，每隔 0.1 秒调用一次 publishMultiResMap() 函数，
// 将全局网格地图中的点云数据发布到 /map/occupancy/global 话题。
void GlobalGridMap::visCallback(const ros::TimerEvent &e)
{
    publishMultiResMap();
}
