//
//  Sliding Occupancy Grid Map (SOGM)
//
//  Created by Yong Li on 2023/4/9.
//

#ifndef SOGMMap_hpp
#define SOGMMap_hpp

#include <stdio.h>
#include <iostream>
#include <memory>
#include <queue>
#include <chrono>
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "RayCast.hpp"

#include <fstream>
#include <cstring>

#include <multilayer/VoxelGridMsg.h>
#include "ObjectPool.hpp"

// 新增头文件，支持多线程处理
#include <thread>      // 用于std::thread
#include <mutex>       // 用于std::mutex, std::lock_guard
#include <vector>      // 用于std::vector (虽然可能已经被隐式包含)
#include <algorithm>   // 用于std::min, std::max
#include <limits>      // 用于std::numeric_limits
#include <atomic>      // 用于原子操作 (替代GCC内置函数的标准替代)
#include <unordered_set>
#include <unordered_map>

#define logit(x) (log((x) / (1 - (x))))

enum class LayerType : uint8_t
{
    BLOCK = 0,
    VOXEL,
    SUBVOXEL,
};

enum class VoxelProjectionStatus {
    BEHIND_CAMERA,      // 在相机后方
    OUT_OF_IMAGE,       // 超出图像范围
    INSUFFICIENT_DATA,  // 深度数据不足
    OCCLUDED,          // 被遮挡
    MATCHED,           // 匹配（在深度图中且深度一致）
    NOT_MATCHED        // 不匹配（在深度图中但深度不一致）

};

class Voxel;    // 前向声明Voxel类
template <typename T>
class ObjectPool; // 前向声明ObjectPool模板类

class Voxel {
public:
    std::vector<float> subvoxel_values_; // 直接存储子体素概率值
    float occupancy_value_;              // 体素整体的占据概率值
    bool is_free_;                  // 是否空闲
    bool is_subvoxel_allocated_;         // 是否已分配子体素

    Voxel(float value) : is_free_(true), is_subvoxel_allocated_(false),
                                         occupancy_value_(value) {
        // 初始不分配子体素内存，按需分配
    }

    void allocate_subvoxels(unsigned int total_subvoxels, const float value) {
        if (!is_subvoxel_allocated_) {
            subvoxel_values_.resize(total_subvoxels, value);
            for(unsigned int i = 0; i < total_subvoxels; ++i) {
                subvoxel_values_[i] = value;
            }
            is_subvoxel_allocated_ = true;
        }
    }
    
    // 释放子体素内存
    void free_subvoxels() {
        if (is_subvoxel_allocated_) {
            std::vector<float>().swap(subvoxel_values_);
            is_subvoxel_allocated_ = false;
        }
    }
    
    // 检查体素是否空闲
    bool is_free() const {
        return is_free_;
    }
};

class Block {
public:
    std::vector<Voxel*> voxels_;         // 体素指针数组 
    float occupancy_value_;              // 块整体的占据概率值
    bool is_voxel_allocated_;            // 是否已分配体素
    bool is_free_;                  // 是否空闲
    LayerType layer_;
    
    // Block的构造和析构函数
    Block() : is_voxel_allocated_(false),
              occupancy_value_(0.0f), 
              layer_(LayerType::BLOCK),
              is_free_(true) {}
    
    // 检查块是否空闲
    bool is_free() const {
        return is_free_;
    }

    void allocate_voxels(unsigned int total_voxels, const float value, ObjectPool<Voxel>& voxel_pool){
        if (!is_voxel_allocated_) {
            voxels_.resize(total_voxels, nullptr);
            for (unsigned int i = 0; i < total_voxels; ++i) {
                voxels_[i] = voxel_pool.acquire(value); // 从池中获取Voxel对象
            }
            is_voxel_allocated_ = true;
        }
    }

    void free_all_voxels(ObjectPool<Voxel>& voxel_pool){
        if (is_voxel_allocated_) {
            for (Voxel* voxel_ptr : voxels_) {
                if (voxel_ptr != nullptr) {
                    voxel_pool.release(voxel_ptr); // 将Voxel对象归还到池中
                }
            }
            voxels_.clear();
            is_voxel_allocated_ = false;
        }
    }
};

static const int NUM_BLOCK_MUTEXES = 32; // 块互斥锁数量
static int file_sequence_number = 0; // 文件序列号，用于文件命名

class SOGMMap
{
private:
    // 存储不同层级体素的数据结构
    struct LayerVoxel {
        Eigen::Vector3d position;  // 体素中心位置
        LayerType layer;           // 层级类型

        LayerVoxel(const Eigen::Vector3d& pos, LayerType lyr)
            : position(pos), layer(lyr) {}
    };

    std::vector<LayerVoxel> new_occupied_voxels_;   // 新占据的多层级体素
    std::vector<LayerVoxel> new_freed_voxels_;      // 新空闲的多层级体素

    enum
    {
        UNKNOWN = -100,
        DISCOVER = 0
    };

    // 声明对象池
    std::unique_ptr<ObjectPool<Block>> block_pool_;
    std::unique_ptr<ObjectPool<Voxel>> voxel_pool_;

    double sub_voxel_res_, voxel_res_, block_res_;
    double sub_voxel_res_inv_, voxel_res_inv_, block_res_inv_;
    int voxel_depth_, block_depth_;
    std::vector<Block*> blocks_;
    int local_subvoxel_mask_;
    int local_voxel_mask_;
    int voxel_to_block_depth_;
    int voxel_depth_double_;
    int voxel_to_block_depth_double_;

    // 用于追踪所有被激活（非空）的Block的线性索引
    std::unordered_set<int> active_block_indices_; 

    double sub_voxel_radius_, voxel_radius_, block_radius_;
    double sub_radius_ratio_, voxel_radius_ratio_, block_radius_ratio_;

    Eigen::Vector3d map_size_;
    Eigen::Vector3i origin_block_;
    Eigen::Vector3i block_num_;
    
    int voxel_num_in_block_, voxel_num_in_block_square_, total_voxel_in_block_;
    int subvoxel_num_in_voxel_, subvoxel_num_in_voxel_square_, total_subvoxel_in_voxel_;

    int block_num_x_, block_num_xy_;
    int block_num_z_, block_num_yz_;

    Eigen::Vector3d camera_pos_;
    double depth_maxdist_, depth_mindist_;
    int skip_pixel_;

    float MIN_VALID_RATIO_SUB_;
    float MIN_VALID_RATIO_VOXEL_;
    float MIN_VALID_RATIO_BLOCK_;

    double MIN_VALID_RATIO_RATIO_;

    double depth_threshold_subvoxel_;
    double depth_threshold_voxel_;
    double depth_threshold_block_;

    // 添加相机内参作为成员变量
    double fx_, fy_, cx_, cy_;
    int depth_width_, depth_height_;
    double k_depth_scaling_factor_;
    double inv_depth_scaling_factor_;
    Eigen::Matrix3d R_C_2_B_;
    Eigen::Vector3d T_C_2_B_;

    std::vector<int> slideClearIndex_;

    float prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;

    // 距离阈值
    double near_distance_threshold_;  // 近距离阈值，使用子体素
    double far_distance_threshold_;   // 远距离阈值，使用块

    // 存储需要贝叶斯概率更新的块索引
    std::queue<int> cache_index_;
    std::vector<char> flag_traverse_, flag_rayend_;
    char raycast_num_;

    // 存储待更新的体素信息
    std::vector<int> occ_blocks_;
    std::vector<int> free_blocks_;
    // 计数器，记录处理的体素数量
    int processed_blocks_count_ = 0;

    
    // 多线程处理相关
    int num_projection_threads_;  // 投影使用的线程数
    std::vector<std::mutex> block_mutex_ = std::vector<std::mutex>(NUM_BLOCK_MUTEXES); // 块互斥锁
    std::mutex active_indices_mutex_;

    // 线程安全的更新函数
    void setCacheOccupancyThreadSafe(int index, int occ_value,  char current_raycast);

    // slide map
    Eigen::Vector3i computeBlockOrigin(const Eigen::Vector3d &new_camera_pos);
    void postShiftBlocks(const Eigen::Vector3i origin_block, const Eigen::Vector3i new_origin_block);
    void getAndClearBlockSlice(const int i, const int width, const int dimension);
    void slideMap(const Eigen::Vector3d camera_pos);
    void resetBlock(int block_idx);

    // help for update occupancy
    
    void beyesProcess();
    
    // 计算点的方位角和仰角（相对于相机坐标系）
    void cartesianToPolar(const Eigen::Vector3d& pt_camera, double& azimuth, double& elevation);
    
    // 计算体素边界的极角范围
    void computeAngularBounds(const Eigen::Vector3d& center_camera, double distance,
                                  double voxel_size, double& min_azimuth, double& max_azimuth,
                                  double& min_elevation, double& max_elevation);
    
    // 极角到像素坐标的转换
    void polarToPixel(double azimuth, double elevation, int& u, int& v);
    // 将此新函数添加到 src/MultiLayerSOGMMap.cpp 中
    bool getDepthInterval(const cv::Mat &depth_image,
                               int min_u, int max_u, int min_v, int max_v,
                               double voxel_z_min, double voxel_z_max,
                               double& sensor_z_min, double& sensor_z_max,
                               double& ratio,
                               double resolution);

    // 多分辨率投影与更新方法
    void switchLayerWithProject(int block_idx, const Eigen::Vector3d& sensor_pos, 
                          const cv::Mat& depth_image,
                          const Eigen::Matrix3d& R_W_2_C,
                          const Eigen::Vector3d& T_W_2_C);
    bool switchLayerWithProjectWithUpdateGlobal(int block_idx, const Eigen::Vector3d& sensor_pos, 
                          const cv::Mat& depth_image,
                          const Eigen::Matrix3d& R_W_2_C,
                          const Eigen::Vector3d& T_W_2_C);
    void raycastProcess(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr, 
        pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr, 
        Eigen::Vector3d camera_pos);
    void voxelPolarProjectionProcessWithRaycast(const cv::Mat &depth_image, 
        const Eigen::Matrix3d &R_C_2_W, 
        const Eigen::Vector3d &T_C_2_W);
    void propagateOccupancyUp(Block* block);
    void propagateOccupancyDown(Block* block, float probability_value);
    void updateOccupancyValue(float &value, bool &is_free, float update);
    VoxelProjectionStatus projectVoxelOnce(const cv::Mat &depth_image, 
                                               const Eigen::Matrix3d &R_W_2_C,
                                               const Eigen::Vector3d &T_W_2_C, 
                                               const Eigen::Vector3d &voxel_center,
                                               const LayerType layer_type,
                                               double resolution,
                                               double valid_ratio,
                                               double depth_threshold);
    double getAdaptiveValidRatio(double distance, LayerType layer) const;
    
    // 辅助函数
    bool createDirectoryIfNotExists(const std::string& path);
public:
    SOGMMap();
    ~SOGMMap();

    void init(std::string filename);
    
    // 设置相机参数
    void setCameraParameters(double fx, double fy, double cx, double cy, 
                            int depth_width, int depth_height,
                            double k_depth_scaling_factor = 1000.0,
                            double depth_maxdist = 100.0,
                            double depth_mindist = 0.1,
                            int skip_pixel = 1,
                            Eigen::Matrix3d R_C_2_B = Eigen::Matrix3d::Identity(),
                            Eigen::Vector3d T_C_2_B = Eigen::Vector3d::Zero());

    void update(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr, pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr,
        const cv::Mat &depth_image, const Eigen::Matrix3d &R_C_2_W,
        const Eigen::Vector3d &T_C_2_W, Eigen::Vector3d camera_pos);

    // main interface
    std::vector<int> *getSlideClearIndex();
    float getOccupancy(const Eigen::Vector3d pos);
    bool isOccupied(const int linear_idx);
    bool isOccupied(const Eigen::Vector3i block_idx);
    bool isOccupied(const Eigen::Vector3d pos);
    bool isOccupiedPrecise(const Eigen::Vector3d& pos);

    bool saveSemanticKittiVoxels(const std::string& base_directory, 
                                      const Eigen::Vector3d& sensor_pos,
                                      double voxel_size,
                                      int grid_x, int grid_y, int grid_z);
            
    bool saveSemanticKittiVoxels(const std::string& base_directory, 
                                      const Eigen::Vector3d& sensor_pos);
    // 序列号管理
    void resetFileSequenceNumber();
    void setFileSequenceNumber(int sequence);

    // 新增：获取局部地图中所有被占据的、多分辨率栅格的状态
    void getLocalMapState(std::vector<multilayer::VoxelGridMsg>& occupied_cells) const;

    // 多分辨接口函数
    const std::vector<LayerVoxel>& getNewOccupiedLayerVoxels() const;
    const std::vector<LayerVoxel>& getNewFreedLayerVoxels() const;
    // 将体素转换为ROS消息
    void fillVoxelGridMsg(std::vector<multilayer::VoxelGridMsg>& msg_array, 
                         const std::vector<LayerVoxel>& voxels) const;

    // get map parameter functions
    double getResolution();
    double getResInv();
    Eigen::Vector3d getSize();
    Eigen::Vector3i getOrigin();
    int getNum();
    Eigen::Vector3i getNum3dim();
    void getBoundary(Eigen::Vector3d &origin, Eigen::Vector3d &size);

    // 世界坐标与各级索引转换
    void worldToSubVoxelIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const;
    void worldToVoxelIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const;
    void worldToBlockIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const;

    // 索引间转换
    void subVoxelIdxToVoxelIdx(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3i& voxel_idx) const;
    void subVoxelIdxToBlockIdx(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3i& block_idx) const;
    void voxelIdxToBlockIdx(const Eigen::Vector3i& voxel_idx, Eigen::Vector3i& block_idx) const;

    // 各级索引转世界坐标（返回中心点）
    void subVoxelIdxToWorld(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3d& pos) const;
    void voxelIdxToWorld(const Eigen::Vector3i& voxel_idx, Eigen::Vector3d& pos) const;
    void blockIdxToWorld(const Eigen::Vector3i& block_idx, Eigen::Vector3d& pos) const;

    // 线性索引转换
    void subVoxelIdxToLocalLinear(const Eigen::Vector3i& subvoxel_idx, int& local_linear_idx) const;
    void voxelIdxToLocalLinear(const Eigen::Vector3i& voxel_idx, int& local_linear_idx) const;
    void blockIdxToLocalLinear(const Eigen::Vector3i& block_idx, int& local_linear_idx) const;

    // 线性索引转三维索引
    Eigen::Vector3i localLinearToSubVoxelIdx(int linear_idx, const Eigen::Vector3i& voxel_idx) const;
    Eigen::Vector3i localLinearToVoxelIdx(int linear_idx, const Eigen::Vector3i& block_idx) const;
    Eigen::Vector3i linearToBlockIdx(int linear_idx) const;

    // 检查索引是否在地图范围内
    bool isSubVoxelIdxInMap(const Eigen::Vector3i& subvoxel_idx) const;
    bool isVoxelIdxInMap(const Eigen::Vector3i& voxel_idx) const;
    bool isBlockIdxInMap(const Eigen::Vector3i& block_idx) const;

    // help functions
    Eigen::Vector3d closetPointInMap(Eigen::Vector3d pos, Eigen::Vector3d camera_pos);

    void setObstacles(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr);
    void clearObstacles(pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr);

    void clearMap();

    typedef std::shared_ptr<SOGMMap> Ptr;
};

#endif
