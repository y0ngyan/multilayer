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

int GlobalGridMap::getSize()
{
    return GlPoints_.size();
}

void GlobalGridMap::addPoint(pcl::PointXYZ pt)
{
    GlPoints_.insert(pt);
}

void GlobalGridMap::addPoints(pcl::PointCloud<pcl::PointXYZ> *ptsPtr)
{
    for (std::size_t i = 0; i < ptsPtr->size(); i++)
        GlPoints_.insert(ptsPtr->at(i));
}

void GlobalGridMap::erasePoint(pcl::PointXYZ pt)
{
    GlPoints_.erase(pt);
}

void GlobalGridMap::erasePoints(pcl::PointCloud<pcl::PointXYZ> *ptsPtr)
{
    for (std::size_t i = 0; i < ptsPtr->size(); i++)
        GlPoints_.erase(ptsPtr->at(i));
}

pcl::PointCloud<pcl::PointXYZ> GlobalGridMap::getMapAll()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    std::set<pcl::PointXYZ, comp>::iterator itr;
    for (itr = GlPoints_.begin(); itr != GlPoints_.end(); itr++)
    {
        cloud.push_back(*itr);
    }

    return cloud;
}

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

    pcl::PointCloud<pcl::PointXYZ>
        cloud = getMapAll();
    pcl::io::savePLYFile(filename, cloud);

    return true;
}

void GlobalGridMap::init(ros::NodeHandle &nh)
{
    new_occ_sub_ = nh.subscribe("/map/new_occ", 1, &GlobalGridMap::newOccCallback, this);
    new_free_sub_ = nh.subscribe("/map/new_free", 10, &GlobalGridMap::newFreeCallback, this);
    save_map_sub_ = nh.subscribe("/map/save", 1, &GlobalGridMap::saveMapCallback, this);

    // 多分辨率
    multi_res_occ_sub_ = nh.subscribe("/map/multi_res_occ", 1, &GlobalGridMap::multiResOccCallback, this);
    multi_res_free_sub_ = nh.subscribe("/map/multi_res_free", 10, &GlobalGridMap::multiResFreeCallback, this);
    multi_res_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/multi_res_global", 10);

    // 添加分层级发布器
    block_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/block", 10);
    voxel_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/voxel", 10);
    subvoxel_layer_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/layer/subvoxel", 10);

    glmap_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map/occupancy/global", 10);

    vis_timer_ = nh.createTimer(ros::Duration(0.1), &GlobalGridMap::visCallback, this);

    // 清空数据结构
    GLblocks_.clear();
    GLvoxels_.clear();
    GLsubvoxels_.clear();

    std::cout << "[GLOBAL MAP] init" << std::endl;
}

// 接收到新的占据点云数据后，调用 addPoints() 函数，
// 将新的占据点云数据添加到全局网格地图中。
void GlobalGridMap::newOccCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    addPoints(&cloud);
}

// 接收到新的自由点云数据后，调用 erasePoints() 函数，
// 将新的自由点云数据从全局网格地图中删除。
void GlobalGridMap::newFreeCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    erasePoints(&cloud);
}

// 处理多分辨率占据体素的回调
void GlobalGridMap::multiResOccCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg) {
    std::vector<MultiResVoxel> blocks;
    std::vector<MultiResVoxel> voxels;
    std::vector<MultiResVoxel> subvoxels;
    blocks.reserve(msg->voxels.size());
    voxels.reserve(msg->voxels.size());
    subvoxels.reserve(msg->voxels.size());
    
    for (const auto& voxel_msg : msg->voxels) {
        MultiResVoxel block;
        MultiResVoxel voxel;
        MultiResVoxel subvoxel;

        switch (voxel_msg.layer) {
            case 0: // BLOCK
                block.layer = voxel_msg.layer;
                block.center.x = voxel_msg.position.x;
                block.center.y = voxel_msg.position.y;
                block.center.z = voxel_msg.position.z;
                block.size = voxel_msg.size;
                block.occupancy = voxel_msg.occupancy_value;
                blocks.push_back(block);
                break;
            case 1: // VOXEL
                voxel.layer = voxel_msg.layer;
                voxel.center.x = voxel_msg.position.x;
                voxel.center.y = voxel_msg.position.y;
                voxel.center.z = voxel_msg.position.z;
                voxel.size = voxel_msg.size;
                voxel.occupancy = voxel_msg.occupancy_value;
                voxels.push_back(voxel);
                break;
            case 2: // SUBVOXEL
                subvoxel.layer = voxel_msg.layer;
                subvoxel.center.x = voxel_msg.position.x;
                subvoxel.center.y = voxel_msg.position.y;
                subvoxel.center.z = voxel_msg.position.z;
                subvoxel.size = voxel_msg.size;
                subvoxel.occupancy = voxel_msg.occupancy_value;
                subvoxels.push_back(subvoxel);
                break;
            default:
                break;
        }
    }
    
    addMultiResVoxels(blocks, voxels, subvoxels);
}

// 处理多分辨率空闲体素的回调
void GlobalGridMap::multiResFreeCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg) {
    std::vector<MultiResVoxel> blocks;
    std::vector<MultiResVoxel> voxels;
    std::vector<MultiResVoxel> subvoxels;
    blocks.reserve(msg->voxels.size());
    voxels.reserve(msg->voxels.size());
    subvoxels.reserve(msg->voxels.size());
    
    for (const auto& voxel_msg : msg->voxels) {
        MultiResVoxel voxel;
        voxel.center.x = voxel_msg.position.x;
        voxel.center.y = voxel_msg.position.y;
        voxel.center.z = voxel_msg.position.z;
        voxel.size = voxel_msg.size;
        voxel.occupancy = voxel_msg.occupancy_value;
        voxel.layer = voxel_msg.layer;
        
        // 根据层级分类
        switch (voxel_msg.layer) {
            case 0: // BLOCK
                blocks.push_back(voxel);
                break;
            case 1: // VOXEL
                voxels.push_back(voxel);
                break;
            case 2: // SUBVOXEL
                subvoxels.push_back(voxel);
                break;
            default:
                break;
        }
    }

    removeMultiResVoxels(blocks, voxels, subvoxels);
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
    block_cloud.header.frame_id = "map";
    voxel_cloud.header.frame_id = "map";
    subvoxel_cloud.header.frame_id = "map";
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

// 将全局网格地图中的点云数据发布到 /map/occupancy/global 话题，
// 以便在 RViz 中可视化显示。
void GlobalGridMap::publishGLmap()
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud = getMapAll();

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    // 指定点云数据的坐标系为 map 坐标系。
    cloud.header.frame_id = "map";

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    glmap_pub_.publish(cloud_msg);
}

// 定时器回调函数，每隔 0.1 秒调用一次 publishGLmap() 函数，
// 将全局网格地图中的点云数据发布到 /map/occupancy/global 话题。
void GlobalGridMap::visCallback(const ros::TimerEvent &e)
{
    // publishGLmap();
    publishMultiResMap();
}
