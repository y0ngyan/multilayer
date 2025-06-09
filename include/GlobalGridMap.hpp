/*
 * @Name:
 * @Author:       yong
 * @Date: 2023-03-12 21:53:01
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-10-22 14:28:42
 * @Description:
 * @Subscriber:
 * @Publisher:
 */

#ifndef GlobalGridMap_hpp
#define GlobalGridMap_hpp

#include <set>
#include <Eigen/Eigen>
#include <time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <multilayer/VoxelGridMsg.h>  
#include <multilayer/VoxelGridMsgArray.h>
#include <unordered_set>

struct comp
{
    bool operator()(const pcl::PointXYZ &pt1, const pcl::PointXYZ &pt2) const
    {
        if (pt1.x < pt2.x)
            return true;
        else if (pt1.x == pt2.x)
        {
            if (pt1.y < pt2.y)
                return true;
            else if (pt1.y == pt2.y)
            {
                if (pt1.z < pt2.z)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
        else
            return false;
    }
};

class GlobalGridMap
{
private:
    // 多级分辨率表示所需的结构
    struct MultiResVoxel {
        pcl::PointXYZ center;    // 体素中心
        // uint8_t layer;           // 层级类型
        
        bool operator==(const MultiResVoxel& other) const {
            return center.x == other.center.x && 
                   center.y == other.center.y && 
                   center.z == other.center.z;
        }
    };

    struct MultiResVoxelHash {
        std::size_t operator()(const MultiResVoxel& voxel) const {
            // 简单哈希函数，确保相同位置大小的体素具有相同哈希
            std::size_t h1 = std::hash<float>()(voxel.center.x);
            std::size_t h2 = std::hash<float>()(voxel.center.y);
            std::size_t h3 = std::hash<float>()(voxel.center.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    // 使用unordered_set保存多分辨率体素
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLblocks_;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLvoxels_;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLsubvoxels_;

    typedef message_filters::sync_policies::ApproximateTime<
        multilayer::VoxelGridMsgArray, 
        multilayer::VoxelGridMsgArray> SyncPolicyVoxelGrids;
    typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyVoxelGrids>> SynchronizerVoxelGrids;
    std::shared_ptr<message_filters::Subscriber<multilayer::VoxelGridMsgArray>> multi_res_occ_sub_sync_;
    std::shared_ptr<message_filters::Subscriber<multilayer::VoxelGridMsgArray>> multi_res_free_sub_sync_;
    SynchronizerVoxelGrids sync_voxel_grids_;

    // ros::Subscriber multi_res_occ_sub_;
    // ros::Subscriber multi_res_free_sub_;
    ros::Publisher multi_res_map_pub_;
    ros::Publisher block_layer_pub_;   // 块级别发布器
    ros::Publisher voxel_layer_pub_;   // 体素级别发布器
    ros::Publisher subvoxel_layer_pub_; // 子体素级别发布器

    ros::Subscriber save_map_sub_;
    ros::Timer vis_timer_;

    std::string frame_id_;  // 添加frame_id成员变量

    // 同步回调函数
    void syncVoxelGridsCallback(const multilayer::VoxelGridMsgArrayConstPtr& occ_msg,
                                const multilayer::VoxelGridMsgArrayConstPtr& free_msg);

    // 多分辨率
    // void multiResOccCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg);
    // void multiResFreeCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg);
    void publishMultiResMap();
    void getLayerMapCloud(pcl::PointCloud<pcl::PointXYZRGB> &block_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB> &voxel_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB> &subvoxel_cloud);
    void publishLayerMaps();

    void saveMapCallback(const std_msgs::BoolConstPtr &msg);

    void visCallback(const ros::TimerEvent &e);

public:
    GlobalGridMap() {};
    ~GlobalGridMap() {};

    // 添加和移除多分辨率体素的方法
    void addMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                           const std::vector<MultiResVoxel>& voxels,
                           const std::vector<MultiResVoxel>& subvoxels);
    void removeMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                              const std::vector<MultiResVoxel>& voxels,
                              const std::vector<MultiResVoxel>& subvoxels);

    // 获取所有多分辨率体素并转换为PCL点云供可视化
    void getMultiResMapCloud(pcl::PointCloud<pcl::PointXYZRGB> &block_cloud,
                             pcl::PointCloud<pcl::PointXYZRGB> &voxel_cloud,
                             pcl::PointCloud<pcl::PointXYZRGB> &subvoxel_cloud);

    bool saveMap();

    void init(ros::NodeHandle &nh);
};

#endif
