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
        float size;              // 体素尺寸
        float occupancy;         // 占据值
        uint8_t layer;           // 层级类型
        
        bool operator==(const MultiResVoxel& other) const {
            return center.x == other.center.x && 
                   center.y == other.center.y && 
                   center.z == other.center.z &&
                   size == other.size;
        }
    };

    struct MultiResVoxelHash {
        std::size_t operator()(const MultiResVoxel& voxel) const {
            // 简单哈希函数，确保相同位置大小的体素具有相同哈希
            std::size_t h1 = std::hash<float>()(voxel.center.x);
            std::size_t h2 = std::hash<float>()(voxel.center.y);
            std::size_t h3 = std::hash<float>()(voxel.center.z);
            std::size_t h4 = std::hash<float>()(voxel.size);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };

    // 使用unordered_set保存多分辨率体素
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> multiResVoxels_;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLblocks_;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLvoxels_;
    std::unordered_set<MultiResVoxel, MultiResVoxelHash> GLsubvoxels_;

    ros::Subscriber multi_res_occ_sub_;
    ros::Subscriber multi_res_free_sub_;
    ros::Publisher multi_res_map_pub_;
    ros::Publisher block_layer_pub_;   // 块级别发布器
    ros::Publisher voxel_layer_pub_;   // 体素级别发布器
    ros::Publisher subvoxel_layer_pub_; // 子体素级别发布器
    // std::set 是一种关联容器，含有 Key 类型对象的已排序集。
    // 用比较函数 比较 (Compare) 进行排序。搜索、移除和插入拥有对数复杂度。
    // set 通常以红黑树实现。
    std::set<pcl::PointXYZ, comp> GlPoints_;

    ros::Publisher glmap_pub_;
    ros::Subscriber new_occ_sub_, new_free_sub_;
    ros::Subscriber save_map_sub_;
    ros::Timer vis_timer_;

    void newOccCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
    void newFreeCallback(const sensor_msgs::PointCloud2ConstPtr &msg);

    // 多分辨率
    void multiResOccCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg);
    void multiResFreeCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg);
    void publishMultiResMap();
    void getLayerMapCloud(pcl::PointCloud<pcl::PointXYZRGB> &block_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB> &voxel_cloud,
                          pcl::PointCloud<pcl::PointXYZRGB> &subvoxel_cloud);
    void publishLayerMaps();

    void saveMapCallback(const std_msgs::BoolConstPtr &msg);

    void publishGLmap();
    void visCallback(const ros::TimerEvent &e);

public:
    GlobalGridMap() {};
    ~GlobalGridMap() {};

    int getSize();

    void addPoint(pcl::PointXYZ pt);
    void addPoints(pcl::PointCloud<pcl::PointXYZ> *ptsPtr);

    // 添加和移除多分辨率体素的方法
    void addMultiResVoxel(const MultiResVoxel& voxel);
    void addMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                           const std::vector<MultiResVoxel>& voxels,
                           const std::vector<MultiResVoxel>& subvoxels);
    void removeMultiResVoxel(const MultiResVoxel& voxel);
    void removeMultiResVoxels(const std::vector<MultiResVoxel>& blocks,
                              const std::vector<MultiResVoxel>& voxels,
                              const std::vector<MultiResVoxel>& subvoxels);

    void erasePoint(pcl::PointXYZ pt);
    void erasePoints(pcl::PointCloud<pcl::PointXYZ> *ptsPtr);

    pcl::PointCloud<pcl::PointXYZ> getMapAll();

    // 获取所有多分辨率体素并转换为PCL点云供可视化
    void getMultiResMapCloud(pcl::PointCloud<pcl::PointXYZRGB> &block_cloud,
                             pcl::PointCloud<pcl::PointXYZRGB> &voxel_cloud,
                             pcl::PointCloud<pcl::PointXYZRGB> &subvoxel_cloud);

    bool saveMap();

    void init(ros::NodeHandle &nh);
};

#endif
