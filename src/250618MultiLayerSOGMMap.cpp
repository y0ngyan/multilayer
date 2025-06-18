#include "MultiLayerSOGMMap.hpp"
#include <set>

SOGMMap::SOGMMap()
{
}

SOGMMap::~SOGMMap()
{
    // 1. 清理所有块并归还到池中
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i] != nullptr) {
            // 先释放块内的所有体素
            blocks_[i]->free_all_voxels(*voxel_pool_);
            // 然后将块归还到池中
            block_pool_->release(blocks_[i]);
            blocks_[i] = nullptr;
        }
    }
    
    // 2. 清空容器
    blocks_.clear();
    active_block_indices_.clear();
    
    std::cout << "[SOGMMap] 析构完成，所有资源已释放" << std::endl;
}

// 从yaml文件中读取地图参数
// 地图分辨率、地图尺寸，计算分辨率导数
// 计算地图体素数量
// 判断地图大小是否过大
// 初始化相机位置为原点
// 初始化光线投射计数器、占用值初始化为unknown
// 初始化命中和命中数计数器、命中计数器、遍历标志位（射线投影是否穿过栅格？？）、光线结束标志位
// 读取概率参数，命中概率、未命中概率、最小概率、最大概率、占用概率
// 计算概率值对数
void SOGMMap::init(std::string filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "无法打开配置文件: " << filename << std::endl;
        return;
    }

    // 读取地图基本参数
    cv::FileNode OccMap_node = fs["OccMap"];
    sub_voxel_res_ = (double)(OccMap_node["resolution"]);
    sub_voxel_res_inv_ = 1.0 / sub_voxel_res_;
    
    // 初始化多分辨率参数
    voxel_depth_ = (int)(OccMap_node["voxel_depth"]);
    block_depth_ = (int)(OccMap_node["block_depth"]);

    near_distance_threshold_ = (double)(OccMap_node["near_distance_threshold"]);
    far_distance_threshold_ = (double)(OccMap_node["far_distance_threshold"]);

    MIN_VALID_RATIO_SUB_ = (float)(OccMap_node["min_valid_ratio_sub"]);
    MIN_VALID_RATIO_VOXEL_ = (float)(OccMap_node["min_valid_ratio_voxel"]);
    MIN_VALID_RATIO_BLOCK_ = (float)(OccMap_node["min_valid_ratio_block"]);

    MIN_VALID_RATIO_RATIO_ = (double)(OccMap_node["min_valid_ratio_ratio"]);

    depth_threshold_subvoxel_ = (double)(OccMap_node["depth_threshold_subvoxel"]);
    depth_threshold_voxel_ = (double)(OccMap_node["depth_threshold_voxel"]);
    depth_threshold_block_ = (double)(OccMap_node["depth_threshold_block_"]);
    
    // 计算各级分辨率和单位数量
    subvoxel_num_in_voxel_ = (1 << voxel_depth_);  // 2^voxel_depth
    subvoxel_num_in_voxel_square_ = subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_;
    total_subvoxel_in_voxel_ = subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_;
    
    voxel_num_in_block_ = (1 << (block_depth_ - voxel_depth_));  // 2^(block_depth - voxel_depth)
    voxel_num_in_block_square_ = voxel_num_in_block_ * voxel_num_in_block_;
    total_voxel_in_block_ = voxel_num_in_block_ * voxel_num_in_block_ * voxel_num_in_block_;

    // 局部坐标变化
    voxel_to_block_depth_ = block_depth_ - voxel_depth_;
    local_subvoxel_mask_ = (1 << voxel_depth_) - 1;
    local_voxel_mask_ = (1 << voxel_to_block_depth_) - 1;
    voxel_depth_double_ = 2 * voxel_depth_;
    voxel_to_block_depth_double_ = 2 * voxel_to_block_depth_;
    
    // 计算不同层级的分辨率
    voxel_res_ = sub_voxel_res_ * subvoxel_num_in_voxel_;
    block_res_ = voxel_res_ * voxel_num_in_block_;
    voxel_res_inv_ = 1.0 / voxel_res_;
    block_res_inv_ = 1.0 / block_res_;

    sub_radius_ratio_ = (double)(OccMap_node["sub_radius_ratio"], 0.5);
    voxel_radius_ratio_ = (double)(OccMap_node["voxel_radius_ratio"], 0.5);
    block_radius_ratio_ = (double)(OccMap_node["block_radius_ratio"], 0.5);
    sub_voxel_radius_ = sub_radius_ratio_ * sub_voxel_res_ * sqrt(3.0);
    voxel_radius_ = voxel_radius_ratio_ * voxel_res_ * sqrt(3.0);
    block_radius_ = block_radius_ratio_ * block_res_ * sqrt(3.0);
    
    // 读取地图尺寸参数
    double map_x = (double)(OccMap_node["map_x"]);
    double map_y = (double)(OccMap_node["map_y"]);
    double map_z = (double)(OccMap_node["map_z"]);
    map_size_ = Eigen::Vector3d(map_x, map_y, map_z);
    
    // 计算地图块数量
    int block_x = std::ceil(map_x / block_res_);
    int block_y = std::ceil(map_y / block_res_);
    int block_z = std::ceil(map_z / block_res_);
    block_num_ = Eigen::Vector3i(block_x, block_y, block_z);
    
    block_num_x_ = block_num_[0];
    block_num_xy_ = block_num_x_ * block_num_[1];
    block_num_z_ = block_num_[2];
    block_num_yz_ = block_num_[1] * block_num_z_;
    
    // 初始化相机位置为原点
    camera_pos_ = Eigen::Vector3d(0, 0, 0);
    // 初始化原点位置 (默认在地图中心)
    origin_block_ = computeBlockOrigin(camera_pos_);

    raycast_num_ = 0;
    
    // 检查地图大小是否合理
    int total_blocks = block_num_[0] * block_num_[1] * block_num_[2];
    if (total_blocks <= 0 || total_blocks > INT_MAX * 0.75) {
        std::cerr << "地图块数量不合理: " << total_blocks << std::endl;
        return;
    }
    std::cout << "[SOGMMap] Creating block pool with " << total_blocks << " blocks" << std::endl;
    block_pool_ = std::make_unique<ObjectPool<Block>>(total_blocks);

    double expected_active_ratio = 0.025; // 预期只有5%的块会是活动的
    double expected_voxel_ratio = 0.05;   // 预期只有10%的体素会被分配
    size_t initial_voxel_pool_size = static_cast<size_t>(
        total_blocks * total_voxel_in_block_ * expected_active_ratio * expected_voxel_ratio
    );
    std::cout << "[SOGMMap] Creating voxel pool with " << initial_voxel_pool_size << " voxels" << std::endl;
    voxel_pool_ = std::make_unique<ObjectPool<Voxel>>(initial_voxel_pool_size);
    
    // 初始化块数组
    std::cout << "[SOGMMap] Initializing " << total_blocks << " blocks" << std::endl;
    blocks_.resize(total_blocks);
    for (int i = 0; i < total_blocks; ++i) {
        // blocks_[i] = block_pool_->acquire(); // 从池中获取Block对象
        try {
            blocks_[i] = block_pool_->acquire();
            if (blocks_[i] == nullptr) {
                std::cerr << "[SOGMMap] ERROR: Failed to acquire block " << i << std::endl;
                return;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SOGMMap] ERROR: Exception when acquiring block " << i << ": " << e.what() << std::endl;
            return;
        }
    }
    std::cout << "[SOGMMap] Successfully initialized all blocks" << std::endl;

    flag_traverse_ = std::vector<char>(total_blocks, 0);
    flag_rayend_ = std::vector<char>(total_blocks, 0);
    
    // 设置多线程投影的线程数
    num_projection_threads_ = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    num_projection_threads_ = std::min(num_projection_threads_, 16);
    std::cout << "[SOGMMap INIT] Using " << num_projection_threads_ << " threads for voxel projection" << std::endl;

    // 读取占据概率参数
    cv::FileNode Prob_node = fs["Probability"];
    double prob_hit = (float)(Prob_node["p_hit"]);
    double prob_miss = (float)(Prob_node["p_miss"]);
    double prob_min = (float)(Prob_node["p_min"]);
    double prob_max = (float)(Prob_node["p_max"]);
    double prob_occupancy = (float)(Prob_node["p_occ"]);

    // 计算对数概率值
    prob_hit_log_ = logit(prob_hit);
    prob_miss_log_ = logit(prob_miss);
    clamp_min_log_ = logit(prob_min);
    clamp_max_log_ = logit(prob_max);
    min_occupancy_log_ = logit(prob_occupancy);

    std::cout << "[SOGMMap] prob_hit: " << prob_hit << ", logit: " << prob_hit_log_ << std::endl;
    std::cout << "[SOGMMap] prob_miss: " << prob_miss << ", logit: " << prob_miss_log_ << std::endl;
    std::cout << "[SOGMMap] prob_min: " << prob_min << ", logit: " << clamp_min_log_ << std::endl;
    std::cout << "[SOGMMap] prob_max: " << prob_max << ", logit: " << clamp_max_log_ << std::endl;
    std::cout << "[SOGMMap] prob_occupancy: " << prob_occupancy << ", logit: " << min_occupancy_log_ << std::endl;

    // 初始化数据结构
    slideClearIndex_.clear();

    // 初始化完成
    std::cout << "[SOGMMap] 初始化完成, 地图尺寸: " << map_x << " x " << map_y << " x " << map_z << " m" << std::endl;
    std::cout << "[SOGMMap] 子体素分辨率: " << sub_voxel_res_ << " m, 体素分辨率: " << voxel_res_ 
              << " m, 块分辨率: " << block_res_ << " m" << std::endl;
    std::cout << "[SOGMMap] 块数量: " << block_num_.transpose() << " 总块数: " << total_blocks << std::endl;
    
    fs.release();
}

void SOGMMap::setCameraParameters(double fx, double fy, double cx, double cy, 
        int depth_width, int depth_height,
        double k_depth_scaling_factor, double depth_maxdist, 
        double depth_mindist, int skip_pixel,
        Eigen::Matrix3d R_C_2_B, Eigen::Vector3d T_C_2_B) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    depth_width_ = depth_width;
    depth_height_ = depth_height;
    k_depth_scaling_factor_ = k_depth_scaling_factor;
    inv_depth_scaling_factor_ = 1.0 / k_depth_scaling_factor_;
    depth_maxdist_ = depth_maxdist;
    depth_mindist_ = depth_mindist;
    skip_pixel_ = skip_pixel;

    R_C_2_B_ = R_C_2_B;
    T_C_2_B_ = T_C_2_B;
}

void SOGMMap::update(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr, pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr,
        const cv::Mat &depth_image, const Eigen::Matrix3d &R_C_2_W,
        const Eigen::Vector3d &T_C_2_W, Eigen::Vector3d camera_pos){

    static int update_count = 0;
    update_count++;
    
    // 每100次更新打印一次内存使用情况
    if (update_count % 100 == 0) {
        auto block_stats = block_pool_->getStats();
        auto voxel_stats = voxel_pool_->getStats();
        
        std::cout << "[SOGMMap] Update #" << update_count 
                  << " - Active blocks: " << active_block_indices_.size()
                  << ", Block pool: " << block_stats.used_count << "/" << block_stats.total_capacity
                  << ", Voxel pool: " << voxel_stats.used_count << "/" << voxel_stats.total_capacity
                  << std::endl;
                  
        // 如果使用率过高，发出警告
        if (voxel_stats.used_count > voxel_stats.total_capacity * 0.9) {
            std::cout << "[SOGMMap] WARNING: Voxel pool usage is very high!" << std::endl;
        }
    }

    slideMap(camera_pos);
    // ros::Time t1 = ros::Time::now();
    raycastProcess(ptws_hit_ptr, ptws_miss_ptr, camera_pos);
    // ros::Time t2 = ros::Time::now();
    // std::cout << "raycast time: " << (t2 - t1).toSec() * 1000 << " ms" << std::endl;
    // t1 = ros::Time::now();
    voxelPolarProjectionProcessWithRaycast(depth_image, R_C_2_W, T_C_2_W);
    // t2 = ros::Time::now();
    // std::cout << "projection time: " << (t2 - t1).toSec() * 1000 << " ms" << std::endl;
}

void SOGMMap::updateGroundBlocks(const pcl::PointCloud<pcl::PointXYZ>& ground_cloud)
{
    updated_ground_block_indices_.clear();

    for (const auto& point : ground_cloud.points)
    {
        Eigen::Vector3d pt_w(point.x, point.y, point.z);

        Eigen::Vector3i block_idx;
        worldToBlockIdx(pt_w, block_idx);

        if (!isBlockIdxInMap(block_idx))
        {
            continue;
        }

        int block_linear_idx;
        blockIdxToLocalLinear(block_idx, block_linear_idx);
        
        // 检查是否已经处理过这个Block
        if (updated_ground_block_indices_.count(block_linear_idx))
        {
            continue;
        }

        // 锁定对应的互斥锁，保证线程安全
        std::lock_guard<std::mutex> lock(block_mutex_[block_linear_idx % NUM_BLOCK_MUTEXES]);

        Block* block = blocks_[block_linear_idx];
        if (block == nullptr) {
            continue;
        }

        // 1. 强制层级为BLOCK：如果之前是更精细的层级，释放其子结构
        if (block->is_voxel_allocated_) {
            block->free_all_voxels(*voxel_pool_);
        }
        block->layer_ = LayerType::BLOCK;

        // 2. 增加占据概率：将地面视为一个坚实的、被占据的表面
        // 这是当前最简单的处理方式，让机器人知道这里不可穿越。
        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
        
        // 3. 标记为已更新
        updated_ground_block_indices_.insert(block_linear_idx);

        // 4. 更新活动索引
        {
            std::lock_guard<std::mutex> active_lock(active_indices_mutex_);
            if (!block->is_free()) {
                active_block_indices_.insert(block_linear_idx);
            } else {
                active_block_indices_.erase(block_linear_idx);
            }
        }
    }
}

// slide map
void SOGMMap::slideMap(const Eigen::Vector3d camera_pos)
{
    slideClearIndex_.clear();

    Eigen::Vector3i new_origin_block;
    Eigen::Vector3i block_shift;

    camera_pos_ = camera_pos;
    new_origin_block = computeBlockOrigin(camera_pos);

    block_shift = new_origin_block - origin_block_;
    if (block_shift(0) == 0 && block_shift(1) == 0 && block_shift(2) == 0)
    {
        return;
    }
    // 处理块级别的滑动
    postShiftBlocks(origin_block_, new_origin_block);
    // 更新原点块
    origin_block_ = new_origin_block;
}

Eigen::Vector3i SOGMMap::computeBlockOrigin(const Eigen::Vector3d &camera_pos)
{
    Eigen::Vector3i new_origin_block;
    // 将相机坐标转换为块索引
    worldToBlockIdx(camera_pos, new_origin_block);
    // 计算原点偏移量，使相机位于地图中心
    for (size_t i = 0; i < 3; i++)
    {
        new_origin_block[i] = new_origin_block[i] - (block_num_[i] >> 1);
    }
    return new_origin_block;
}

void SOGMMap::postShiftBlocks(const Eigen::Vector3i origin_block, const Eigen::Vector3i new_origin_block)
{
    Eigen::Vector3i clear_width;
    Eigen::Vector3i block_shift;
    block_shift = new_origin_block - origin_block;
    // 计算每个维度需要清除的块数量
    for (size_t i = 0; i < 3; i++)
    {
        clear_width[i] = std::min(abs(block_shift[i]), block_num_[i]);
    }
    // 对每个维度进行处理
    for (size_t i = 0; i < 3; i++)
    {
        if (block_shift[i] > 0)
        {
            // 正向滑动，清除前方的块
            getAndClearBlockSlice(0, clear_width[i], i);
        }
        else if (block_shift[i] < 0)
        {
            // 负向滑动，清除后方的块
            getAndClearBlockSlice(block_num_[i] - clear_width[i], clear_width[i], i);
        }
    }
}

void SOGMMap::getAndClearBlockSlice(const int i, const int width, const int dimension)
{
    // set minimum dimensions
    int ixyz_min[3] = {0, 0, 0};
    ixyz_min[dimension] = i;

    // set max dimensions
    int ixyz_max[3] = {block_num_(0), block_num_(1), block_num_(2)};
    ixyz_max[dimension] = i + width;

    Eigen::Vector3i ixyz;
    int local_linear_idx;
    for (int ix = ixyz_min[0]; ix < ixyz_max[0]; ix++)
    {
        for (int iy = ixyz_min[1]; iy < ixyz_max[1]; iy++)
        {
            for (int iz = ixyz_min[2]; iz < ixyz_max[2]; iz++)
            {
                ixyz(0) = ix + origin_block_(0);
                ixyz(1) = iy + origin_block_(1);
                ixyz(2) = iz + origin_block_(2);

                blockIdxToLocalLinear(ixyz, local_linear_idx);
                // 重置该块
                resetBlock(local_linear_idx);
                slideClearIndex_.push_back(local_linear_idx);
            }
        }
    }
}

void SOGMMap::resetBlock(int block_idx)
{
    // 检查块索引是否有效
    if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
        return;
    }
    Block* block = blocks_[block_idx];
    // 确保块指针有效
    if (block == nullptr) {
        return;
    }
    // **** 核心改动：将内部的 Voxel 归还到池中 ****
    block->free_all_voxels(*voxel_pool_);
    // 重置块的状态
    block->occupancy_value_ = 0.0f;
    block->is_free_ = true;
    block->layer_ = LayerType::BLOCK;

    // 从活动索引中移除这个块
    active_block_indices_.erase(block_idx);
}

// main interface
std::vector<int> *SOGMMap::getSlideClearIndex()
{
    return &slideClearIndex_;
}

const std::vector<SOGMMap::LayerVoxel>& SOGMMap::getNewOccupiedLayerVoxels() const {
    return new_occupied_voxels_;
}

// 获取新空闲的多层级体素
const std::vector<SOGMMap::LayerVoxel>& SOGMMap::getNewFreedLayerVoxels() const {
    return new_freed_voxels_;
}

float SOGMMap::getOccupancy(const Eigen::Vector3d pos)
{
    Eigen::Vector3i block_idx;
    worldToBlockIdx(pos, block_idx);
    int index;
    blockIdxToLocalLinear(block_idx, index);
    if (index < 0 || index >= blocks_.size() || blocks_[index] == nullptr)
    {
        return -1.0; // Invalid block index
    }
    Block *block = blocks_[index];
    return block->occupancy_value_;
}

bool SOGMMap::isOccupied(const Eigen::Vector3i block_idx)
{
    int index;
    blockIdxToLocalLinear(block_idx, index);
    return isOccupied(index);
}

bool SOGMMap::isOccupied(const Eigen::Vector3d pos)
{
    Eigen::Vector3i block_idx;
    worldToBlockIdx(pos, block_idx);
    int index;
    blockIdxToLocalLinear(block_idx, index);
    return isOccupied(index);
}

bool SOGMMap::isOccupied(const int index)
{
    if (index < 0 || index >= blocks_.size() || blocks_[index] == nullptr)
    {
        return false; // Invalid block index
    }
    Block *block = blocks_[index];
    return !block->is_free();
}

bool SOGMMap::isOccupiedPrecise(const Eigen::Vector3d& pos)
{
    // 1. 先检查块级别
    Eigen::Vector3i block_idx;
    worldToBlockIdx(pos, block_idx);
    int block_linear_idx;
    blockIdxToLocalLinear(block_idx, block_linear_idx);
    
    if (block_linear_idx < 0 || block_linear_idx >= blocks_.size() || blocks_[block_linear_idx] == nullptr)
    {
        return false; // Invalid block index
    }
    
    Block *block = blocks_[block_linear_idx];
    
    // 如果块是空闲的，直接返回false
    if (block->is_free()) {
        return false;
    }
    
    // 2. 根据块的层级进行精确检查
    switch (block->layer_) {
        case LayerType::BLOCK: {
            // 块级别：直接返回块的占据状态
            return !block->is_free();
        }
        
        case LayerType::VOXEL: {
            if (!block->is_voxel_allocated_) {
                return !block->is_free();
            }
            
            // 体素级别：检查对应体素的占据状态
            Eigen::Vector3i voxel_idx;
            worldToVoxelIdx(pos, voxel_idx);
            int local_voxel_idx;
            voxelIdxToLocalLinear(voxel_idx, local_voxel_idx);
            
            if (local_voxel_idx >= 0 && local_voxel_idx < block->voxels_.size() && 
                block->voxels_[local_voxel_idx] != nullptr) {
                return !block->voxels_[local_voxel_idx]->is_free();
            }
            return false;
        }
        
        case LayerType::SUBVOXEL: {
            if (!block->is_voxel_allocated_) {
                return !block->is_free();
            }
            
            // 子体素级别：检查对应子体素的占据状态
            Eigen::Vector3i voxel_idx;
            worldToVoxelIdx(pos, voxel_idx);
            int local_voxel_idx;
            voxelIdxToLocalLinear(voxel_idx, local_voxel_idx);
            
            if (local_voxel_idx >= 0 && local_voxel_idx < block->voxels_.size() && 
                block->voxels_[local_voxel_idx] != nullptr) {
                
                Voxel* voxel = block->voxels_[local_voxel_idx];
                if (!voxel->is_subvoxel_allocated_) {
                    return !voxel->is_free();
                }
                
                // 检查子体素
                Eigen::Vector3i subvoxel_idx;
                worldToSubVoxelIdx(pos, subvoxel_idx);
                int local_subvoxel_idx;
                subVoxelIdxToLocalLinear(subvoxel_idx, local_subvoxel_idx);
                
                if (local_subvoxel_idx >= 0 && local_subvoxel_idx < voxel->subvoxel_values_.size()) {
                    return voxel->subvoxel_values_[local_subvoxel_idx] >= min_occupancy_log_;
                }
            }
            return false;
        }
    }
    
    return false;
}

// get map parameter functions
double SOGMMap::getResolution()
{
    return block_res_;
}

double SOGMMap::getResInv()
{
    return block_res_inv_;
}

Eigen::Vector3d SOGMMap::getSize()
{
    return map_size_;
}

Eigen::Vector3i SOGMMap::getOrigin()
{
    return origin_block_;
}

int SOGMMap::getNum()
{
    return block_num_[0] * block_num_[1] * block_num_[2];
}

Eigen::Vector3i SOGMMap::getNum3dim()
{
    return block_num_;
}

void SOGMMap::getBoundary(Eigen::Vector3d &origin, Eigen::Vector3d &size)
{
    blockIdxToWorld(origin_block_, origin);
    size = map_size_;
}

// help functions
Eigen::Vector3d SOGMMap::closetPointInMap(Eigen::Vector3d pos, Eigen::Vector3d camera_pos)
{
    Eigen::Vector3d diff = pos - camera_pos;

    Eigen::Vector3d map_max_boundary;
    Eigen::Vector3d map_min_boundary;
    blockIdxToWorld(origin_block_, map_min_boundary);
    blockIdxToWorld(origin_block_ + block_num_, map_max_boundary);

    Eigen::Vector3d max_tc = map_max_boundary - camera_pos;
    Eigen::Vector3d min_tc = map_min_boundary - camera_pos;
    double min_t = 1000000;

    for (size_t i = 0; i < 3; ++i)
    {
        if (fabs(diff[i]) > 0)
        {

            double t1 = max_tc[i] / diff[i];
            if (t1 > 0 && t1 < min_t)
                min_t = t1;

            double t2 = min_tc[i] / diff[i];
            if (t2 > 0 && t2 < min_t)
                min_t = t2;
        }
    }

    return camera_pos + (min_t - 0.05) * diff;
}


void SOGMMap::raycastProcess(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr, pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr, Eigen::Vector3d camera_pos)
{
    raycast_num_ += 1;
    char current_raycast = static_cast<char>(raycast_num_ & 0xFF);
    
    // 确保标志数组大小与地图块数量匹配
    if (flag_traverse_.size() != getNum()) {
        flag_traverse_ = std::vector<char>(getNum(), 0);
    }
    if (flag_rayend_.size() != getNum()) {
        flag_rayend_ = std::vector<char>(getNum(), 0);
    }

    Eigen::Vector3d ray_end = camera_pos * block_res_inv_;
    
    // 清空之前收集的块
    occ_blocks_.clear();
    free_blocks_.clear();
    
    // 为防止线程冲突，每个线程使用独立的缓存
    struct ThreadCache {
        std::vector<int> local_occ_blocks;
        std::vector<int> local_free_blocks;
        int processed_count = 0;
    };
    
    // 计算点云分块处理策略
    size_t total_points = ptws_hit_ptr->size() + ptws_miss_ptr->size();
    int points_per_thread = (total_points + num_projection_threads_ - 1) / num_projection_threads_;
    
    // 创建线程并分配工作
    std::vector<std::thread> threads;
    std::vector<ThreadCache> thread_caches(num_projection_threads_);
    
    // 为每个线程的缓存预分配合理空间
    for (auto& cache : thread_caches) {
        cache.local_occ_blocks.reserve(points_per_thread);
        cache.local_free_blocks.reserve(points_per_thread * 25); // 光线可能穿过多个空闲块
    }
    
    // ros::Time t1 = ros::Time::now();
    for (int t = 0; t < num_projection_threads_; ++t) {
        size_t hit_start = t * points_per_thread;
        size_t hit_end = std::min(hit_start + points_per_thread, ptws_hit_ptr->size());
        
        size_t miss_start = 0;
        size_t miss_end = 0;
        
        // 如果hit部分分配完毕，继续分配miss部分
        if (hit_end == ptws_hit_ptr->size()) {
            size_t remaining = points_per_thread - (hit_end - hit_start);
            miss_start = (t - (total_points - ptws_miss_ptr->size()) / points_per_thread) * points_per_thread;
            miss_end = std::min(miss_start + remaining, ptws_miss_ptr->size());
        }
        
        // 创建线程，使用引用捕获避免数据复制
        threads.emplace_back([&, t, hit_start, hit_end, miss_start, miss_end, current_raycast]() {
            RayCaster raycaster;
            ThreadCache& cache = thread_caches[t];
            
            // 1. 处理命中点云
            for (size_t i = hit_start; i < hit_end; ++i) {
                // 获取点的世界坐标
                Eigen::Vector3d pt_w{ptws_hit_ptr->at(i).x, ptws_hit_ptr->at(i).y, ptws_hit_ptr->at(i).z};
                
                // 将世界坐标转换为块索引
                Eigen::Vector3i block_idx;
                worldToBlockIdx(pt_w, block_idx);
                
                if (isBlockIdxInMap(block_idx)) {
                    int block_linear_idx;
                    blockIdxToLocalLinear(block_idx, block_linear_idx);
                    
                    // 使用原子操作检查和设置flag_rayend_标志
                    bool already_marked = false;
                    #pragma omp atomic read
                    already_marked = (flag_rayend_[block_linear_idx] == current_raycast);

                    if (!already_marked) {
                        #pragma omp atomic write
                        flag_rayend_[block_linear_idx] = current_raycast;
                        cache.local_occ_blocks.push_back(block_linear_idx);
                    }
                    
                    // 检查是否已处理过该块的光线投射
                    #pragma omp atomic read
                    already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                    
                    if (already_marked) {
                        continue; // 如果已处理，则跳过此点
                    }
                    
                    #pragma omp atomic write
                    flag_traverse_[block_linear_idx] = current_raycast;
                    cache.processed_count++;
                } else {
                    // 如果点不在地图中，找到地图边界附近的点
                    pt_w = closetPointInMap(pt_w, camera_pos);
                    worldToBlockIdx(pt_w, block_idx);
                }
                
                // 创建从命中点到相机的光线
                Eigen::Vector3d ray_start = pt_w * block_res_inv_;
                
                // 光线步进参数化，使用块分辨率
                raycaster.setInput(ray_start, ray_end);
                
                // 收集光线经过的所有块
                Eigen::Vector3i ray_block;
                while (raycaster.step(ray_block)) {
                    
                    int block_linear_idx;
                    blockIdxToLocalLinear(ray_block, block_linear_idx);

                    if (updated_ground_block_indices_.count(block_linear_idx)) {
                        continue;
                    }
                    
                    // 标记未命中的块（光线穿过但不是终点）
                    bool already_marked = false;
                    #pragma omp atomic read
                    already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                    
                    if (already_marked) {
                        // break; // 如果已处理过，则跳过
                        continue;
                    }
                    
                    #pragma omp atomic write
                    flag_traverse_[block_linear_idx] = current_raycast;
                    cache.processed_count++;
                    
                    // 将光线经过的块添加到空闲列表
                    cache.local_free_blocks.push_back(block_linear_idx);
                }
            }
            
            // 2. 处理未命中点云
            for (size_t i = miss_start; i < miss_end; ++i) {
                // 同样获取点的世界坐标
                Eigen::Vector3d pt_w{ptws_miss_ptr->at(i).x, ptws_miss_ptr->at(i).y, ptws_miss_ptr->at(i).z};
                
                // 将世界坐标转换为块索引
                Eigen::Vector3i block_idx;
                worldToBlockIdx(pt_w, block_idx);
                
                if (!isBlockIdxInMap(block_idx)) {
                    pt_w = closetPointInMap(pt_w, camera_pos);
                    worldToBlockIdx(pt_w, block_idx);
                }
                
                int block_linear_idx;
                blockIdxToLocalLinear(block_idx, block_linear_idx);
                
                // 将未命中的终点块标记为空闲
                bool already_marked = false;
                bool is_ground_block = false;
                #pragma omp atomic read
                already_marked = (flag_traverse_[block_linear_idx] == current_raycast);

                #pragma omp atomic read
                is_ground_block = updated_ground_block_indices_.count(block_linear_idx);
                
                if (!already_marked || !is_ground_block) {
                    #pragma omp atomic write
                    flag_traverse_[block_linear_idx] = current_raycast;
                    cache.processed_count++;
                    cache.local_free_blocks.push_back(block_linear_idx);
                }
                
                // 从未命中点到相机的光线投射
                Eigen::Vector3d ray_start = pt_w * block_res_inv_;
                
                // 同样使用块分辨率进行光线投射
                raycaster.setInput(ray_start, ray_end);
                
                // 处理光线经过的所有块
                Eigen::Vector3i ray_block;
                while (raycaster.step(ray_block)) {
                    blockIdxToLocalLinear(ray_block, block_linear_idx);

                    if (updated_ground_block_indices_.count(block_linear_idx)) {
                        continue;
                    }
                    
                    bool already_marked = false;
                    #pragma omp atomic read
                    already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                    
                    if (already_marked) {
                        // break; // 如果已处理过，则跳过
                        continue;
                    }
                    
                    #pragma omp atomic write
                    flag_traverse_[block_linear_idx] = current_raycast;
                    cache.processed_count++;
                    
                    // 将光线经过的块添加到空闲列表
                    cache.local_free_blocks.push_back(block_linear_idx);
                }
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // ros::Time t2 = ros::Time::now();
    // std::cout << "raycast time: " << (t2 - t1).toSec() * 1000 << " ms" << std::endl;

    // 合并所有线程的结果
    int total_processed_count = 0;
    
    // 预先计算结果大小，避免多次重新分配内存
    size_t total_occ_size = 0;
    size_t total_free_size = 0;
    for (const auto& cache : thread_caches) {
        total_occ_size += cache.local_occ_blocks.size();
        total_free_size += cache.local_free_blocks.size();
        total_processed_count += cache.processed_count;
    }
    
    // 预先分配空间
    occ_blocks_.reserve(total_occ_size);
    free_blocks_.reserve(total_free_size);

    // t1 = ros::Time::now();
    
    // 合并结果，去除重复项
    std::unordered_set<int> unique_occ_blocks;
    std::unordered_set<int> unique_free_blocks;
    
    for (const auto& cache : thread_caches) {
        for (const auto& block_idx : cache.local_occ_blocks) {
            unique_occ_blocks.insert(block_idx);
        }
        
        for (const auto& block_idx : cache.local_free_blocks) {
            // 如果块已经在占用列表中，不添加到自由列表
            if (unique_occ_blocks.find(block_idx) == unique_occ_blocks.end()) {
                unique_free_blocks.insert(block_idx);
            }
        }
    }
    
    // 将去重后的结果复制到最终列表
    occ_blocks_.insert(occ_blocks_.end(), unique_occ_blocks.begin(), unique_occ_blocks.end());
    free_blocks_.insert(free_blocks_.end(), unique_free_blocks.begin(), unique_free_blocks.end());
    
    // 更新处理的块数量统计
    processed_blocks_count_ = total_processed_count;
}

double SOGMMap::getAdaptiveValidRatio(double distance, LayerType layer) const {
    double ratio = 1.0; // 默认值
    switch (layer) {
        case LayerType::SUBVOXEL:
            ratio = 0.9 * exp(-0.15 * distance) + 0.05;
            return std::max(ratio, MIN_VALID_RATIO_RATIO_);  
        case LayerType::VOXEL:
            if (distance <= near_distance_threshold_) {
                ratio = exp(-0.2 * distance);
            }
            else {
                ratio = 0.75 * exp(-0.1 * (distance - far_distance_threshold_));
            }
            return std::max(ratio, MIN_VALID_RATIO_RATIO_);
        case LayerType::BLOCK:
            ratio = 0.75 * exp(-0.1 * distance);
            return std::max(ratio, MIN_VALID_RATIO_RATIO_);
        default:
            return 1.0;
    }
}

VoxelProjectionStatus SOGMMap::projectVoxelOnce(const cv::Mat &depth_image, 
                                               const Eigen::Matrix3d &R_W_2_C,
                                               const Eigen::Vector3d &T_W_2_C, 
                                               const Eigen::Vector3d &voxel_center,
                                               const LayerType layer_type,
                                               double resolution,
                                               double valid_ratio,
                                               double depth_threshold) {
    // === 步骤 1: 计算体素在相机坐标系下的深度区间 ===
    Eigen::Vector3d center_camera = R_W_2_C * voxel_center + T_W_2_C;

    // 如果体素中心在相机后方，直接返回
    if (center_camera.z() <= 0) {
        return VoxelProjectionStatus::BEHIND_CAMERA;
    }

    // 使用解析法计算深度区间
    double half_res = resolution / 2.0;
    const Eigen::RowVector3d& Rz_row = R_W_2_C.row(2);
    double extent = half_res * (Rz_row.lpNorm<1>());
    double voxel_z_min = center_camera.z() - extent;
    double voxel_z_max = center_camera.z() + extent;
    
    // === 步骤 2: 计算投影范围并检查是否在图像内 ===
    double distance = center_camera.norm();
    double radius = half_res * sqrt(3.0);
    double min_azimuth, max_azimuth, min_elevation, max_elevation;
    computeAngularBounds(center_camera, distance, radius, min_azimuth, max_azimuth, min_elevation, max_elevation);

    valid_ratio = valid_ratio * getAdaptiveValidRatio(distance, layer_type);
    
    int min_u, min_v, max_u, max_v;
    polarToPixel(min_azimuth, min_elevation, min_u, min_v);
    polarToPixel(max_azimuth, max_elevation, max_u, max_v);
    
    if (min_u >= max_u || min_v >= max_v) {
        return VoxelProjectionStatus::OUT_OF_IMAGE;
    }

    if (layer_type == LayerType::SUBVOXEL && 
        (min_u < 0 || max_u >= depth_width_ || 
         min_v < 0 || max_v >= depth_height_)) {
        return VoxelProjectionStatus::OUT_OF_IMAGE;
    }

    // 限制像素坐标在深度图尺寸内
    min_u = std::max(0, std::min(min_u, depth_width_ - 1));
    max_u = std::max(0, std::min(max_u, depth_width_ - 1));
    min_v = std::max(0, std::min(min_v, depth_height_ - 1));
    max_v = std::max(0, std::min(max_v, depth_height_ - 1));

    // === 步骤 3: 获取传感器深度区间 ===
    double sensor_z_min, sensor_z_max;
    double ratio = 0.0;
    
    if (!getDepthInterval(depth_image, min_u, max_u, min_v, max_v, voxel_z_min, voxel_z_max, 
                         sensor_z_min, sensor_z_max, ratio, resolution)) {
        if (std::max(voxel_z_min, sensor_z_min) > std::min(voxel_z_max, sensor_z_max) + depth_threshold) {
            return VoxelProjectionStatus::NOT_MATCHED;
        }
        return VoxelProjectionStatus::INSUFFICIENT_DATA;
    }

    if (ratio < valid_ratio) {
        return VoxelProjectionStatus::LESS_VALID_DATA;
    }

    // === 步骤 4: 同时判断遮挡和匹配状态 ===
    
    // 判断是否被遮挡：体素完全在传感器测量值之前
    if (voxel_z_min > sensor_z_max + depth_threshold) {
        return VoxelProjectionStatus::OCCLUDED;
    }
    
    // 判断是否匹配：体素深度区间与传感器深度区间有重叠
    if (std::max(voxel_z_min, sensor_z_min) <= std::min(voxel_z_max, sensor_z_max) + depth_threshold) {
        return VoxelProjectionStatus::MATCHED;
    }
    
    // 否则就是不匹配
    return VoxelProjectionStatus::NOT_MATCHED;
}

// 修改：更新占据值的辅助函数，使其既可以增加也可以减少
void SOGMMap::updateOccupancyValue(float &value, bool &is_free, float update) {
    // 直接使用update值（可以是正值或负值）
    float new_value = value + update;
    float clamped_value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
    
    // 更新free状态
    is_free = (clamped_value < min_occupancy_log_);
    value = clamped_value;
}

// 向上传递概率：从子级到父级更新占据概率
void SOGMMap::propagateOccupancyUp(Block* block) {
    if (block == nullptr) {
        return;
    }
    
    // 如果没有分配体素，无需更新
    if (!block->is_voxel_allocated_) {
        return;
    }
    
    // 重置块状态为空闲，然后检查体素
    block->is_free_ = true;
    float total_occupancy = 0.0f;
    int occupied_count = 0;
    
    // 遍历所有体素
    for (int i = 0; i < block->voxels_.size(); ++i) {
        Voxel* voxel = block->voxels_[i];
        if (voxel == nullptr) {
            continue;
        }
        
        // 如果体素有子体素，先更新体素的占据状态
        if (voxel->is_subvoxel_allocated_) {
            voxel->is_free_ = true;
            float voxel_total_occupancy = 0.0f;
            int voxel_occupied_count = 0;
            
            // 计算子体素的聚合占据信息
            for (const auto& subvoxel_value : voxel->subvoxel_values_) {
                if (subvoxel_value > min_occupancy_log_) {  // 使用阈值判断是否被占据
                    voxel_total_occupancy += subvoxel_value;
                    voxel_occupied_count++;
                }
            }
            
            // 更新体素占据信息
            if (voxel_occupied_count > 0) {
                voxel->is_free_ = false;
                voxel->occupancy_value_ = voxel_total_occupancy / voxel_occupied_count;
            } else {
                voxel->occupancy_value_ = 0.0f;
            }
        }
        
        // 检查体素是否被占据，更新块的统计信息
        if (!voxel->is_free_) {
            block->is_free_ = false;
            total_occupancy += voxel->occupancy_value_;
            occupied_count++;
        }
    }
    
    // 更新块的占据概率
    if (occupied_count > 0) {
        block->occupancy_value_ = total_occupancy / occupied_count;
    } else {
        block->occupancy_value_ = 0.0f;
    }
}

// 向下传递概率：将父级的占据概率传递到子级
void SOGMMap::propagateOccupancyDown(Block* block, float probability_value) {
    if (block == nullptr) {
        return;
    }
    
    // 更新块的占据信息
    if (probability_value >= min_occupancy_log_) {
        block->is_free_ = false;
        block->occupancy_value_ = probability_value;
    } else {
        block->is_free_ = true;
        block->occupancy_value_ = probability_value;
    }
    
    // 如果没有分配体素，不需要向下传递
    if (!block->is_voxel_allocated_) {
        return;
    }
    
    // 向下传递到所有体素
    for (int i = 0; i < block->voxels_.size(); ++i) {
        Voxel* voxel = block->voxels_[i];
        if (voxel == nullptr) {
            continue;
        }
        
        // 更新体素的占据信息
        if (probability_value >= min_occupancy_log_) {
            voxel->is_free_ = false;
            voxel->occupancy_value_ = probability_value;
        } else {
            voxel->is_free_ = true;
            voxel->occupancy_value_ = probability_value;
        }
        
        // 如果体素有子体素，继续向下传递
        if (voxel->is_subvoxel_allocated_) {
            for (int j = 0; j < voxel->subvoxel_values_.size(); ++j) {
                voxel->subvoxel_values_[j] = probability_value;
            }
        }
    }
}

void SOGMMap::switchLayerWithProject(int block_idx, const Eigen::Vector3d& sensor_pos, 
                          const cv::Mat& depth_image,
                          const Eigen::Matrix3d& R_W_2_C,
                          const Eigen::Vector3d& T_W_2_C) {
    // 检查块索引是否有效
    if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
        return;
    }
    
    Block* block = blocks_[block_idx];
    
    // 计算块的中心位置
    Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
    Eigen::Vector3d block_center;
    blockIdxToWorld(block_grid_idx, block_center);
    
    // 计算块中心到传感器的距离
    double distance = (block_center - sensor_pos).norm();
    
    // 根据距离决定使用哪个层级
    LayerType target_layer;
    
    if (distance > far_distance_threshold_) {
        // 远距离：使用块级别表示
        target_layer = LayerType::BLOCK;
    } else if (distance > near_distance_threshold_) {
        // 中等距离：使用体素级别表示
        target_layer = LayerType::VOXEL;
    } else {
        // 近距离：使用子体素级别表示
        target_layer = LayerType::SUBVOXEL;
    }
    
    // 如果目标层级与当前层级相同，无需更改
    if (block->layer_ >= target_layer) {
        return;
    }
    
    // 存储当前占据值，用于向下传递
    float current_occupancy = block->occupancy_value_;
    bool is_occupied = !block->is_free_;
    LayerType old_layer = block->layer_;
    
    // 根据目标层级切换表示
    switch (target_layer) {
        case LayerType::BLOCK: {
            block->free_all_voxels(*voxel_pool_);
            block->layer_ = LayerType::BLOCK;
            break;
        }
        
        case LayerType::VOXEL: {
            if (block->layer_ == LayerType::BLOCK) {
                block->allocate_voxels(total_voxel_in_block_, 0.0f, *voxel_pool_);
                
                for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                    Voxel* voxel = block->voxels_[vox_idx];
                    if (voxel == nullptr) continue;
                    
                    // 获取体素的世界坐标
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                    Eigen::Vector3d voxel_center;
                    voxelIdxToWorld(voxel_grid_idx, voxel_center);

                    // 块是占用的，只有在深度图视野中的体素继承块占据概率
                    VoxelProjectionStatus projection_status = 
                        projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, voxel_center, LayerType::VOXEL, voxel_res_, MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                    if (projection_status == VoxelProjectionStatus::MATCHED) {
                        voxel->occupancy_value_ = current_occupancy;
                        voxel->is_free_ = block->is_free();
                    } 
                    else {
                        // 如果不匹配或不可见，设置为默认值
                        voxel->occupancy_value_ = 0.0f;
                        voxel->is_free_ = true;
                    }
                }
            } 
            else if (block->layer_ == LayerType::SUBVOXEL) {
                // 从子体素级别降级到体素级别
                // 对每个体素，聚合其子体素的占据信息
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    if (block->voxels_[i] != nullptr && block->voxels_[i]->is_subvoxel_allocated_) {
                        // 释放子体素内存
                        block->voxels_[i]->free_subvoxels();
                    }
                }
            }
            
            // 设置层级
            block->layer_ = LayerType::VOXEL;
            break;
        }
        
        case LayerType::SUBVOXEL: {
            if (!block->is_voxel_allocated_) {
                // 如果还没有体素，先分配体素
                block->allocate_voxels(total_voxel_in_block_, 0.0f, *voxel_pool_);
                
                for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                    Voxel* voxel = block->voxels_[vox_idx];
                    if (voxel == nullptr) continue;
                    
                    // 获取体素的世界坐标
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                    Eigen::Vector3d voxel_center;
                    voxelIdxToWorld(voxel_grid_idx, voxel_center);
                    // 块是占用的，只有在深度图视野中的体素继承块占据概率
                    VoxelProjectionStatus projection_status = 
                        projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, voxel_center, LayerType::VOXEL, voxel_res_, MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                    if (projection_status == VoxelProjectionStatus::MATCHED) {
                        voxel->occupancy_value_ = current_occupancy;
                        voxel->is_free_ = block->is_free();
                    } 
                    else {
                        // 如果不匹配或不可见，设置为默认值
                        voxel->occupancy_value_ = 0.0f;
                        voxel->is_free_ = true;
                    }
                }
            }
            
            // 为每个体素分配子体素
            for (int i = 0; i < block->voxels_.size(); ++i) {
                Voxel* voxel = block->voxels_[i];
                if (voxel == nullptr) continue;
                
                float voxel_occupancy = voxel->occupancy_value_;
                bool voxel_is_free = voxel->is_free_;
                
                if (!voxel->is_subvoxel_allocated_) {
                    voxel->allocate_subvoxels(total_subvoxel_in_voxel_, 0.0f);
                    
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(i, block_grid_idx);
                    
                    for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                        Eigen::Vector3d subvoxel_center;
                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                        VoxelProjectionStatus projection_status = 
                            projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, subvoxel_center, LayerType::SUBVOXEL, sub_voxel_res_, MIN_VALID_RATIO_SUB_, depth_threshold_subvoxel_);
                        if (projection_status == VoxelProjectionStatus::MATCHED) {
                            // 如果子体素在深度图中匹配，继承体素的占据概率
                            voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                        } 
                        else {
                            // 如果不匹配或不可见，设置为默认值
                            voxel->subvoxel_values_[subvox_idx] = 0.0f;
                        }
                    }
                }
            }
            // 设置层级并更新块状态
            block->layer_ = LayerType::SUBVOXEL;
            break;
        }
    }

    block = blocks_[block_idx];
    block_grid_idx = linearToBlockIdx(block_idx);
    blockIdxToWorld(block_grid_idx, block_center);
}

bool SOGMMap::switchLayerWithProjectWithUpdateGlobal(int block_idx, const Eigen::Vector3d& sensor_pos, 
                          const cv::Mat& depth_image,
                          const Eigen::Matrix3d& R_W_2_C,
                          const Eigen::Vector3d& T_W_2_C) {
    // 检查块索引是否有效
    if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
        return false;
    }
    
    Block* block = blocks_[block_idx];
    
    // 计算块的中心位置
    Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
    Eigen::Vector3d block_center;
    blockIdxToWorld(block_grid_idx, block_center);
    
    // 计算块中心到传感器的距离
    double distance = (block_center - sensor_pos).norm();
    
    // 根据距离决定使用哪个层级
    LayerType target_layer;
    
    if (distance > far_distance_threshold_) {
        // 远距离：使用块级别表示
        target_layer = LayerType::BLOCK;
    } else if (distance > near_distance_threshold_) {
        // 中等距离：使用体素级别表示
        target_layer = LayerType::VOXEL;
    } else {
        // 近距离：使用子体素级别表示
        target_layer = LayerType::SUBVOXEL;
    }
    
    // 如果目标层级与当前层级相同，无需更改
    if (block->layer_ >= target_layer) {
        return true;
    }
    
    // 存储当前占据值和层级信息，用于记录变化
    float current_occupancy = block->occupancy_value_;
    bool is_occupied = !block->is_free_;
    LayerType old_layer = block->layer_;

    bool block_need_update = false;
    // 根据目标层级切换表示
    switch (target_layer) {
        case LayerType::BLOCK: {
            // 从更精细的表示转为块级别
            if (block->is_voxel_allocated_) {
                block->free_all_voxels(*voxel_pool_);
            }
            block->layer_ = LayerType::BLOCK;
            updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
            break;
        }
        
        case LayerType::VOXEL: {
            if (block->layer_ == LayerType::BLOCK) {
                // 从块级别升级到体素级别
                // 分配体素并将块的占据信息向下传递
                block->allocate_voxels(total_voxel_in_block_, current_occupancy, *voxel_pool_);

                for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                    Voxel* voxel = block->voxels_[vox_idx];
                    if (voxel == nullptr) continue;
                    
                    // 获取体素的世界坐标
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                    Eigen::Vector3d voxel_center;
                    voxelIdxToWorld(voxel_grid_idx, voxel_center);

                    VoxelProjectionStatus projection_status = 
                            projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, voxel_center, LayerType::VOXEL,voxel_res_, MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                    if (block->is_free()) {
                        if (projection_status != VoxelProjectionStatus::OUT_OF_IMAGE) {
                            // 块是空闲的，只有在深度图中的体素才继承块概率
                            voxel->occupancy_value_ = current_occupancy;
                            block_need_update = true;
                        } 
                        else {
                            // 如果不匹配或不可见，设置为默认值
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }
                    else {
                        if (projection_status == VoxelProjectionStatus::MATCHED) {
                            // 如果体素在深度图中匹配，继承块的占据概率
                            voxel->occupancy_value_ = current_occupancy;
                            voxel->is_free_ = block->is_free();
                            updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
                            block_need_update = true;
                        } 
                        else {
                            // 如果不匹配或不可见，设置为默认值
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }      
                }
            } 
            else if (block->layer_ == LayerType::SUBVOXEL) {
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    if (block->voxels_[i] != nullptr && block->voxels_[i]->is_subvoxel_allocated_) {               
                        block->voxels_[i]->free_subvoxels();
                    }
                }
            }
            
            // 设置层级
            block->layer_ = LayerType::VOXEL;
            break;
        }
        
        case LayerType::SUBVOXEL: {
            if (!block->is_voxel_allocated_) {
                // 如果还没有体素，先分配体素
                block->allocate_voxels(total_voxel_in_block_, current_occupancy, *voxel_pool_);

                for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                    Voxel* voxel = block->voxels_[vox_idx];
                    if (voxel == nullptr) continue;
                    
                    // 获取体素的世界坐标
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                    Eigen::Vector3d voxel_center;
                    voxelIdxToWorld(voxel_grid_idx, voxel_center);

                    VoxelProjectionStatus projection_status = 
                            projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, voxel_center, LayerType::VOXEL, voxel_res_, MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                    
                    if (block->is_free()) {
                        if (projection_status != VoxelProjectionStatus::OUT_OF_IMAGE) {
                            // 块是空闲的，只有在深度图中的体素才继承块概率
                            voxel->occupancy_value_ = current_occupancy;
                            voxel->is_free_ = block->is_free();
                        } 
                        else {
                            // 如果不匹配或不可见，设置为默认值
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }
                    else{
                        if (projection_status == VoxelProjectionStatus::MATCHED) {
                            // 如果体素在深度图中匹配，继承块的占据概率
                            voxel->occupancy_value_ = current_occupancy;
                            voxel->is_free_ = block->is_free();
                            updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
                            block_need_update = true;
                        } 
                        else {
                            // 如果不匹配或不可见，设置为默认值
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }
                }
            }
            
            // 为每个体素分配子体素
            for (int i = 0; i < block->voxels_.size(); ++i) {
                Voxel* voxel = block->voxels_[i];
                if (voxel == nullptr) continue;
                
                float voxel_occupancy = voxel->occupancy_value_;
                bool voxel_is_free = voxel->is_free_;
                
                if (!voxel->is_subvoxel_allocated_) {
                    voxel->allocate_subvoxels(total_subvoxel_in_voxel_, voxel_occupancy);
                    
                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(i, block_grid_idx);
                    
                    for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                        Eigen::Vector3d subvoxel_center;
                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

                        VoxelProjectionStatus projection_status = 
                                projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, subvoxel_center, LayerType::SUBVOXEL, sub_voxel_res_, MIN_VALID_RATIO_SUB_, depth_threshold_subvoxel_);
                        
                        if (voxel_is_free) {
                            if (projection_status != VoxelProjectionStatus::OUT_OF_IMAGE) {
                                // 块是空闲的，只有在深度图中的子体素才继承体素概率
                                voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                            } 
                            else {
                                voxel->subvoxel_values_[subvox_idx] = 0.0f;
                            }
                        }
                        else{
                            // 块是占用的，只有投影到深度图上实际值和测量值匹配的才继承体素概率
                            if (projection_status == VoxelProjectionStatus::MATCHED) {
                                voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                                float &value = voxel->subvoxel_values_[subvox_idx];
                                float new_value = value + prob_hit_log_;
                                value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                block_need_update = true;
                            } 
                            else {
                                voxel->subvoxel_values_[subvox_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
            // 设置层级并更新块状态
            block->layer_ = LayerType::SUBVOXEL;
            break;
        }
        if (block_need_update) {
            propagateOccupancyUp(block);
        }
    }

    return false;
}
void SOGMMap::voxelPolarProjectionProcessWithRaycast(const cv::Mat &depth_image, 
                                                  const Eigen::Matrix3d &R_C_2_W, 
                                                  const Eigen::Vector3d &T_C_2_W) {
    Eigen::Matrix3d R_W_2_C = R_C_2_W.transpose();
    Eigen::Vector3d T_W_2_C = -R_W_2_C * T_C_2_W;

    // 1. 处理自由空间块 - 多线程
    int free_count = free_blocks_.size();
    int blocks_per_thread = (free_count + num_projection_threads_ - 1) / num_projection_threads_;
    
    std::vector<std::thread> free_threads;
    free_threads.reserve(num_projection_threads_);
    
    for (int t = 0; t < num_projection_threads_; ++t) {
        size_t start_idx = t * blocks_per_thread;
        size_t end_idx = std::min(start_idx + blocks_per_thread, static_cast<size_t>(free_count));
        
        // 如果这个线程没有要处理的块，则跳过
        if (start_idx >= end_idx) {
            continue;
        }
        
        free_threads.emplace_back([this,
            &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int block_idx = free_blocks_[i];
                
                // 快速检查块索引有效性，无需加锁
                if (block_idx < 0 || block_idx >= blocks_.size()) {
                    continue;
                }
                
                Block* block = nullptr;
                Eigen::Vector3i block_grid_idx;
                Eigen::Vector3d block_center;
                
                // 第一次加锁：获取块指针和基本信息
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    block = blocks_[block_idx];
                    if (block == nullptr) {
                        continue;
                    }
                    
                    // 获取块的世界坐标
                    block_grid_idx = linearToBlockIdx(block_idx);
                    blockIdxToWorld(block_grid_idx, block_center);
                }
                
                // 无锁区域：进行投影计算（这些是纯计算，不涉及共享数据修改）
                VoxelProjectionStatus block_status = VoxelProjectionStatus::NOT_MATCHED;
                bool should_continue = false;
                
                // 检查块是否应该处理（无锁读取block状态进行初步判断）
                if (block->is_free()) {
                    block_status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                    block_center, LayerType::BLOCK, block_res_, 
                                                    MIN_VALID_RATIO_BLOCK_, depth_threshold_block_);
                    if (block_status != VoxelProjectionStatus::MATCHED) {
                        should_continue = true;
                    }
                }
                
                // 第二次加锁：更新块状态
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    
                    // 重新检查块是否仍然有效
                    if (blocks_[block_idx] != block || block == nullptr) {
                        continue;
                    }
                    
                    if (should_continue) {
                        block->free_all_voxels(*voxel_pool_);
                        block->layer_ = LayerType::BLOCK;
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
                        
                        // 更新活动索引
                        {
                            std::lock_guard<std::mutex> active_lock(active_indices_mutex_);
                            if (!block->is_free()) {
                                active_block_indices_.insert(block_idx);
                            } else {
                                active_block_indices_.erase(block_idx);
                            }
                        }
                        continue;
                    }
                }
                
                // 第三次加锁：执行层级切换和更新
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    
                    // 再次验证块状态
                    if (blocks_[block_idx] != block || block == nullptr) {
                        continue;
                    }
                    
                    // 执行层级切换（这个函数内部可能需要进一步优化）
                    switchLayerWithProject(block_idx, T_C_2_W, depth_image, R_W_2_C, T_W_2_C);
                    
                    // 重新获取块引用
                    block = blocks_[block_idx];
                    if (block == nullptr) {
                        continue;
                    }
                }
                
                // 无锁区域：准备体素/子体素的投影计算
                std::vector<VoxelProjectionResult> voxel_results;
                std::vector<SubVoxelProjectionResult> subvoxel_results;
                
                // 根据层级进行无锁的投影计算
                if (block->layer_ == LayerType::VOXEL && block->is_voxel_allocated_) {
                    voxel_results.reserve(block->voxels_.size());
                    
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        if (block->voxels_[vox_idx] != nullptr) {
                            Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                            Eigen::Vector3d voxel_center;
                            voxelIdxToWorld(voxel_grid_idx, voxel_center);
                            
                            VoxelProjectionStatus status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                        voxel_center, LayerType::VOXEL, voxel_res_, 
                                                                        MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                            
                            voxel_results.push_back({vox_idx, status});
                        }
                    }
                } else if (block->layer_ == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
                    subvoxel_results.reserve(total_voxel_in_block_ * total_subvoxel_in_voxel_);
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
                            Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                            Eigen::Vector3d voxel_center;
                            voxelIdxToWorld(voxel_grid_idx, voxel_center);
                            
                            VoxelProjectionStatus voxel_status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                                voxel_center, LayerType::VOXEL, voxel_res_, 
                                                                                MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                            
                            if (voxel_status != VoxelProjectionStatus::OUT_OF_IMAGE) {
                                for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                                    Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                                    Eigen::Vector3d subvoxel_center;
                                    subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                                    
                                    VoxelProjectionStatus status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                                subvoxel_center, LayerType::SUBVOXEL, sub_voxel_res_, 
                                                                                MIN_VALID_RATIO_SUB_, depth_threshold_subvoxel_);
                                    
                                    subvoxel_results.push_back({vox_idx, subvox_idx, status});
                                }
                            }
                        }
                    }
                }
                
                // 第四次加锁：应用计算结果
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    
                    // 最后一次验证块状态
                    if (blocks_[block_idx] != block || block == nullptr) {
                        continue;
                    }
                    
                    // 应用计算结果
                    if (block->layer_ == LayerType::BLOCK) {
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
                    } else if (block->layer_ == LayerType::VOXEL) {
                        for (const auto& result : voxel_results) {
                            if (result.index < block->voxels_.size() && block->voxels_[result.index] != nullptr) {
                                Voxel* voxel = block->voxels_[result.index];
                                switch (result.status) {
                                    case VoxelProjectionStatus::MATCHED:
                                    case VoxelProjectionStatus::OCCLUDED:
                                    case VoxelProjectionStatus::BEHIND_CAMERA:
                                    case VoxelProjectionStatus::OUT_OF_IMAGE:                                    
                                    case VoxelProjectionStatus::LESS_VALID_DATA:
                                        continue;
                                    case VoxelProjectionStatus::NOT_MATCHED:
                                    case VoxelProjectionStatus::INSUFFICIENT_DATA:
                                    default:
                                        updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_miss_log_);
                                        break;
                                }
                            }
                        }
                        propagateOccupancyUp(block);
                    } else if (block->layer_ == LayerType::SUBVOXEL) {
                        for (const auto& result : subvoxel_results) {
                            if (result.voxel_index < block->voxels_.size() && 
                                block->voxels_[result.voxel_index] != nullptr &&
                                result.subvoxel_index < block->voxels_[result.voxel_index]->subvoxel_values_.size()) {
                                
                                float &value = block->voxels_[result.voxel_index]->subvoxel_values_[result.subvoxel_index];
                                float new_value = 0.0f;
                                switch (result.status) {
                                    case VoxelProjectionStatus::MATCHED:
                                        new_value = value + prob_hit_log_;
                                        value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                        break;
                                    case VoxelProjectionStatus::OCCLUDED:
                                    case VoxelProjectionStatus::BEHIND_CAMERA:
                                    case VoxelProjectionStatus::OUT_OF_IMAGE:                                    
                                    case VoxelProjectionStatus::LESS_VALID_DATA:
                                        continue;
                                    case VoxelProjectionStatus::NOT_MATCHED:
                                    case VoxelProjectionStatus::INSUFFICIENT_DATA:
                                    default:
                                        new_value = value + prob_miss_log_;
                                        value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                        break;
                                }
                            }
                        }
                        propagateOccupancyUp(block);
                    }
                    
                    // 更新活动索引
                    {
                        std::lock_guard<std::mutex> active_lock(active_indices_mutex_);
                        if (!block->is_free()) {
                            active_block_indices_.insert(block_idx);
                        } else {
                            active_block_indices_.erase(block_idx);
                        }
                    }
                }
            }
        });
    }
    
    // 等待所有自由空间线程完成
    for (auto& thread : free_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // 2. 处理占据块部分 - 类似的优化
    int occ_count = occ_blocks_.size();
    blocks_per_thread = (occ_count + num_projection_threads_ - 1) / num_projection_threads_;
    
    std::vector<std::thread> occ_threads;
    occ_threads.reserve(num_projection_threads_);
    
    for (int t = 0; t < num_projection_threads_; ++t) {
        size_t start_idx = t * blocks_per_thread;
        size_t end_idx = std::min(start_idx + blocks_per_thread, static_cast<size_t>(occ_count));
        
        if (start_idx >= end_idx) {
            continue;
        }
        
        occ_threads.emplace_back([this,
            &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int block_idx = occ_blocks_[i];
                
                if (block_idx < 0 || block_idx >= blocks_.size()) {
                    continue;
                }
                
                Block* block = nullptr;
                Eigen::Vector3i block_grid_idx;
                Eigen::Vector3d block_center;
                bool was_free = false;
                
                // 第一次加锁：获取块信息并执行层级切换
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    block = blocks_[block_idx];
                    if (block == nullptr) {
                        continue;
                    }
                    
                    was_free = block->is_free_;
                    block_grid_idx = linearToBlockIdx(block_idx);
                    blockIdxToWorld(block_grid_idx, block_center);
                    
                    bool need_update = switchLayerWithProjectWithUpdateGlobal(block_idx, T_C_2_W, depth_image, R_W_2_C, T_W_2_C);
                    
                    if (!need_update) {
                        {
                            std::lock_guard<std::mutex> active_lock(active_indices_mutex_);
                            if (!block->is_free()) {
                                active_block_indices_.insert(block_idx);
                            } else {
                                active_block_indices_.erase(block_idx);
                            }
                        }
                        continue;
                    }
                    
                    // 重新获取块引用
                    block = blocks_[block_idx];
                    if (block == nullptr) {
                        continue;
                    }
                }
                
                // 无锁区域：进行投影计算
                std::vector<VoxelProjectionResult> voxel_results;
                std::vector<SubVoxelProjectionResult> subvoxel_results;
                bool block_updated = false;
                
                if (block->layer_ == LayerType::VOXEL && block->is_voxel_allocated_) {
                    voxel_results.reserve(block->voxels_.size());
                    
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        if (block->voxels_[vox_idx] != nullptr) {
                            Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                            Eigen::Vector3d voxel_center;
                            voxelIdxToWorld(voxel_grid_idx, voxel_center);
                            
                            VoxelProjectionStatus status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                        voxel_center, LayerType::VOXEL, voxel_res_, 
                                                                        MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                            
                            voxel_results.push_back({vox_idx, status});
                        }
                    }
                } else if (block->layer_ == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
                    subvoxel_results.reserve(total_voxel_in_block_ * total_subvoxel_in_voxel_);
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
                            Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                            Eigen::Vector3d voxel_center;
                            voxelIdxToWorld(voxel_grid_idx, voxel_center);
                            
                            VoxelProjectionStatus voxel_status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                                voxel_center, LayerType::VOXEL, voxel_res_, 
                                                                                MIN_VALID_RATIO_VOXEL_, depth_threshold_voxel_);
                            
                            if (voxel_status == VoxelProjectionStatus::MATCHED || voxel_status == VoxelProjectionStatus::LESS_VALID_DATA) {
                                for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                                    Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                                    Eigen::Vector3d subvoxel_center;
                                    subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                                    
                                    VoxelProjectionStatus status = projectVoxelOnce(depth_image, R_W_2_C, T_W_2_C, 
                                                                                subvoxel_center, LayerType::SUBVOXEL, sub_voxel_res_, 
                                                                                MIN_VALID_RATIO_SUB_, depth_threshold_subvoxel_);
                                    
                                    subvoxel_results.push_back({vox_idx, subvox_idx, status});
                                }
                            }
                        }
                    }
                }
                
                // 第二次加锁：应用计算结果
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    
                    // 验证块状态
                    if (blocks_[block_idx] != block || block == nullptr) {
                        continue;
                    }
                    
                    // 应用计算结果
                    if (block->layer_ == LayerType::BLOCK) {
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
                    } else if (block->layer_ == LayerType::VOXEL) {
                        for (const auto& result : voxel_results) {
                            if (result.index < block->voxels_.size() && block->voxels_[result.index] != nullptr) {
                                Voxel* voxel = block->voxels_[result.index];
                                switch (result.status) {
                                    case VoxelProjectionStatus::BEHIND_CAMERA:
                                    case VoxelProjectionStatus::OUT_OF_IMAGE:
                                    case VoxelProjectionStatus::OCCLUDED:                                    
                                    case VoxelProjectionStatus::LESS_VALID_DATA:
                                    case VoxelProjectionStatus::INSUFFICIENT_DATA:
                                        continue;
                                    case VoxelProjectionStatus::NOT_MATCHED:                    
                                        updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_miss_log_);
                                        break;
                                    case VoxelProjectionStatus::MATCHED:
                                        updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
                                        block_updated = true;
                                        break;
                                }
                            }
                        }
                        
                        if (block_updated) {
                            propagateOccupancyUp(block);
                        }
                    } else if (block->layer_ == LayerType::SUBVOXEL) {
                        for (const auto& result : subvoxel_results) {
                            if (result.voxel_index < block->voxels_.size() && 
                                block->voxels_[result.voxel_index] != nullptr &&
                                result.subvoxel_index < block->voxels_[result.voxel_index]->subvoxel_values_.size()) {
                                
                                float &value = block->voxels_[result.voxel_index]->subvoxel_values_[result.subvoxel_index];
                                switch (result.status) {
                                    case VoxelProjectionStatus::BEHIND_CAMERA:
                                    case VoxelProjectionStatus::OUT_OF_IMAGE:
                                    case VoxelProjectionStatus::OCCLUDED:                                    
                                    case VoxelProjectionStatus::LESS_VALID_DATA:
                                    case VoxelProjectionStatus::INSUFFICIENT_DATA:
                                        continue;
                                    case VoxelProjectionStatus::NOT_MATCHED:
                                        {
                                            float new_value = value + prob_miss_log_;
                                            value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                            block_updated = true;
                                        }
                                        break;
                                    case VoxelProjectionStatus::MATCHED:
                                        {
                                            float new_value = value + prob_hit_log_;
                                            value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                            block_updated = true;
                                        }
                                        break;
                                }
                            }
                        }
                        
                        if (block_updated) {
                            propagateOccupancyUp(block);
                        }
                    }
                    
                    // 更新活动索引
                    {
                        std::lock_guard<std::mutex> active_lock(active_indices_mutex_);
                        if (!block->is_free()) {
                            active_block_indices_.insert(block_idx);
                        } else {
                            active_block_indices_.erase(block_idx);
                        }
                    }
                }
            }
        });
    }
    
    // 等待所有占据块线程完成
    for (auto& thread : occ_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // 清空处理过的块列表
    free_blocks_.clear();
    occ_blocks_.clear();
}

// 将多层级体素转换为ROS消息
void SOGMMap::fillVoxelGridMsg(std::vector<multilayer::VoxelGridMsg>& msg_array, 
                             const std::vector<LayerVoxel>& voxels) const {
    msg_array.clear();
    msg_array.reserve(voxels.size());
    
    for (const auto& voxel : voxels) {
        multilayer::VoxelGridMsg msg;
        msg.position.x = voxel.position.x();
        msg.position.y = voxel.position.y();
        msg.position.z = voxel.position.z();
        msg.layer = static_cast<uint8_t>(voxel.layer);
        
        msg_array.push_back(msg);
    }
}

// 计算点的方位角和仰角（相对于相机坐标系）
void SOGMMap::cartesianToPolar(const Eigen::Vector3d& pt_camera, double& azimuth, double& elevation) {
    // 确保点在相机前方
    if (pt_camera.z() <= 0) {
        azimuth = 0;
        elevation = 0;
        return;
    }
    
    // 计算方位角（相对于z轴的水平角度）
    azimuth = atan2(pt_camera.x(), pt_camera.z());
    
    // 计算仰角（相对于xz平面的垂直角度）
    double dist_xz = sqrt(pt_camera.x() * pt_camera.x() + pt_camera.z() * pt_camera.z());
    elevation = atan2(pt_camera.y(), dist_xz);
}

// 极角到像素坐标的转换
void SOGMMap::polarToPixel(double azimuth, double elevation, int& u, int& v) {
    // 根据针孔相机模型，角度与像素坐标的关系
    u = static_cast<int>(cx_ + fx_ * tan(azimuth) + 0.5);
    v = static_cast<int>(cy_ + fy_ * tan(elevation) / cos(azimuth) + 0.5);
}

// 计算体素边界的极角范围
void SOGMMap::computeAngularBounds(const Eigen::Vector3d& center_camera, double distance,
                                       double radius, double& min_azimuth, double& max_azimuth,
                                       double& min_elevation, double& max_elevation) {
    // 计算体素在相机坐标系下的视角范围
    double angular_size = atan2(radius, distance);
    
    // 计算中心点的方位角和仰角
    double center_azimuth, center_elevation;
    cartesianToPolar(center_camera, center_azimuth, center_elevation);
    
    // 根据距离和体素大小计算角度范围
    // 使用余弦定理可以获得更精确的角度范围，这里使用简化近似
    min_azimuth = center_azimuth - angular_size;
    max_azimuth = center_azimuth + angular_size;
    min_elevation = center_elevation - angular_size;
    max_elevation = center_elevation + angular_size;
    
    // 确保方位角在[-π, π]范围内
    while (min_azimuth < -M_PI) min_azimuth += 2 * M_PI;
    while (max_azimuth > M_PI) max_azimuth -= 2 * M_PI;
}

// 替换 src/MultiLayerSOGMMap.cpp 中的 getDepthInterval 函数
bool SOGMMap::getDepthInterval(const cv::Mat &depth_image,
                               int min_u, int max_u, int min_v, int max_v,
                               double voxel_z_min, double voxel_z_max,
                               double& sensor_z_min, double& sensor_z_max,
                               double& ratio,
                               double resolution) {
    // 步骤 1: (保持不变) 执行高效的自适应采样
    int patch_width = max_u - min_u + 1;
    int patch_height = max_v - min_v + 1;
    int total_pixels_in_patch = patch_width * patch_height;
    
    // 根据总体素数量自适应选择采样比例
    double sampling_ratio;
    if (total_pixels_in_patch <= 50) {
        sampling_ratio = 1.0; // 极小范围：全采样
    } else if (total_pixels_in_patch <= 100) {
        sampling_ratio = 0.64;  // 小范围：高采样率
    } else if (total_pixels_in_patch <= 500) {
        sampling_ratio = 0.32;  // 中等范围：中等采样率
    } else if (total_pixels_in_patch <= 1000) {
        sampling_ratio = 0.16; // 大范围：低采样率
    } else {
        sampling_ratio = 0.08; // 超大范围：很低采样率
    }
    
    // 计算采样步长，确保至少采样一些点
    int target_sample_count = static_cast<int>(total_pixels_in_patch * sampling_ratio);
    target_sample_count = std::max(target_sample_count, 9); // 至少采样9个点（3x3网格）
    
    // 根据目标采样数量计算步长
    double step_factor = sqrt(static_cast<double>(total_pixels_in_patch) / target_sample_count);
    int step_u = std::max(1, static_cast<int>(patch_width / sqrt(target_sample_count) + 0.5));
    int step_v = std::max(1, static_cast<int>(patch_height / sqrt(target_sample_count) + 0.5));
    
    // 确保步长不会导致采样点过少
    step_u = std::min(step_u, patch_width);
    step_v = std::min(step_v, patch_height);

    // std::cout << "total pixels: " << (patch_width / step_u) * (patch_height / step_v) << std::endl;

    int total_pixels = 0;
    int valid_pixels = 0;
    bool found_valid_depth = false;

    for (int v = min_v; v <= max_v; v += step_v) {
        if (v < 0 || v >= depth_height_) continue;
        const uint16_t* row_ptr = depth_image.ptr<uint16_t>(v);
        for (int u = min_u; u <= max_u; u += step_u) {
            if (u < 0 || u >= depth_width_) continue;
            double pixel_depth = static_cast<double>(row_ptr[u]) * inv_depth_scaling_factor_;
            if (pixel_depth > depth_mindist_ && pixel_depth < depth_maxdist_) {
                if (!found_valid_depth) {
                    found_valid_depth = true;
                    sensor_z_min = pixel_depth;
                    sensor_z_max = pixel_depth;
                }
                else {
                    sensor_z_min = std::min(sensor_z_min, pixel_depth);
                    sensor_z_max = std::max(sensor_z_max, pixel_depth);
                }
            }
            total_pixels++;
            if (pixel_depth > voxel_z_min && pixel_depth < voxel_z_max) {
                valid_pixels++;
            }
        }
    }

    // 为了进行分布分析，我们需要足够多的样本点
    // if (total_pixels == 0) {
    //     return false;
    // }
    if (valid_pixels == 0 ) {
        // std::cout<< "Not enough valid pixels for depth interval calculation." << std::endl;
        // std::cout<< "z_min: " << sensor_z_min << ", z_max: " << sensor_z_max << std::endl;
        return false;
    }

    ratio = static_cast<double>(valid_pixels) / total_pixels;

    return true;
}

// 世界坐标与各级索引转换
void SOGMMap::worldToSubVoxelIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const{
    for (int i = 0; i < 3; i++) {
    idx[i] = static_cast<int>(std::floor(pos[i] * sub_voxel_res_inv_));
}
}
void SOGMMap::worldToVoxelIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const{
    for (int i = 0; i < 3; i++) {
        idx[i] = static_cast<int>(std::floor(pos[i] * voxel_res_inv_));
    }
}
void SOGMMap::worldToBlockIdx(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const{
    for (int i = 0; i < 3; i++) {
        idx[i] = static_cast<int>(std::floor(pos[i] * block_res_inv_));
    }
}

// 索引间转换
void SOGMMap::subVoxelIdxToVoxelIdx(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3i& voxel_idx) const{
    for (int i = 0; i < 3; i++) {
        voxel_idx[i] = subvoxel_idx[i] >> voxel_depth_; 
    }
}
void SOGMMap::subVoxelIdxToBlockIdx(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3i& block_idx) const{
    for (int i = 0; i < 3; i++) {
        block_idx[i] = subvoxel_idx[i] >> block_depth_;
    }
}
void SOGMMap::voxelIdxToBlockIdx(const Eigen::Vector3i& voxel_idx, Eigen::Vector3i& block_idx) const{
    for (int i = 0; i < 3; i++) {
        block_idx[i] = voxel_idx[i] >> (block_depth_ - voxel_depth_); 
    }
}

// 各级索引转世界坐标（返回中心点）
void SOGMMap::subVoxelIdxToWorld(const Eigen::Vector3i& subvoxel_idx, Eigen::Vector3d& pos) const{
    for (int i = 0; i < 3; i++) {
        pos[i] = (subvoxel_idx[i] + 0.5) * sub_voxel_res_;
    }
}
void SOGMMap::voxelIdxToWorld(const Eigen::Vector3i& voxel_idx, Eigen::Vector3d& pos) const{
    for (int i = 0; i < 3; i++) {
        pos[i] = (voxel_idx[i] + 0.5) * voxel_res_;
    }
}
void SOGMMap::blockIdxToWorld(const Eigen::Vector3i& block_idx, Eigen::Vector3d& pos) const{
    for (int i = 0; i < 3; i++) {
        pos[i] = (block_idx[i] + 0.5) * block_res_;
    }
}
void SOGMMap::subVoxelIdxToLocalLinear(const Eigen::Vector3i& subvoxel_idx, int& local_linear_idx) const{
    local_linear_idx = (subvoxel_idx.x() & local_subvoxel_mask_) + 
                       ((subvoxel_idx.y() & local_subvoxel_mask_) << voxel_depth_) + 
                       ((subvoxel_idx.z() & local_subvoxel_mask_) << voxel_depth_double_);
}
void SOGMMap::voxelIdxToLocalLinear(const Eigen::Vector3i& voxel_idx, int& local_linear_idx) const{
    local_linear_idx = (voxel_idx.x() & local_voxel_mask_) + 
                       ((voxel_idx.y() & local_voxel_mask_) << voxel_to_block_depth_) + 
                       ((voxel_idx.z() & local_voxel_mask_) << voxel_to_block_depth_double_);
}
void SOGMMap::blockIdxToLocalLinear(const Eigen::Vector3i& block_idx, int& local_linear_idx) const{
    // 计算块在地图范围内的相对位置
    Eigen::Vector3i block_offset = block_idx;
    for (int i = 0; i < 3; i++) {
        // 处理循环索引
        block_offset[i] = block_offset[i] % block_num_[i];
        if (block_offset[i] < 0) {
            block_offset[i] += block_num_[i];
        }
    }
    
    // 转换为线性索引
    local_linear_idx = block_offset[0]  + block_offset[1] * block_num_x_ + block_offset[2] * block_num_xy_;
    // local_linear_idx = block_offset[2] + block_offset[1] * block_num_z_ + block_offset[0] * block_num_yz_;
}

// 局部线性索引转子体素索引
Eigen::Vector3i SOGMMap::localLinearToSubVoxelIdx(int linear_idx, const Eigen::Vector3i& voxel_idx) const {
    Eigen::Vector3i local_idx;
    // local_idx[2] = linear_idx % subvoxel_num_in_voxel_;
    // linear_idx /= subvoxel_num_in_voxel_;
    // local_idx[1] = linear_idx % subvoxel_num_in_voxel_;
    // local_idx[0] = linear_idx / subvoxel_num_in_voxel_;
    local_idx[0] = linear_idx & local_subvoxel_mask_;  // 提取 x 分量
    local_idx[1] = (linear_idx >> voxel_depth_) & local_subvoxel_mask_;  // 提取 y 分量
    local_idx[2] = (linear_idx >> voxel_depth_double_) & local_subvoxel_mask_;  // 提取 z 分量
    
    // 计算全局子体素索引
    Eigen::Vector3i subvoxel_idx;
    for (int i = 0; i < 3; i++) {
        subvoxel_idx[i] = (voxel_idx[i] << voxel_depth_) + local_idx[i];
    }
    
    return subvoxel_idx;
}

// 局部线性索引转体素索引
Eigen::Vector3i SOGMMap::localLinearToVoxelIdx(int linear_idx, const Eigen::Vector3i& block_idx) const {
    Eigen::Vector3i local_idx;
    // local_idx[2] = linear_idx % voxel_num_in_block_;
    // linear_idx /= voxel_num_in_block_;
    // local_idx[1] = linear_idx % voxel_num_in_block_;
    // local_idx[0] = linear_idx / voxel_num_in_block_;
    local_idx[0] = linear_idx & local_voxel_mask_;  // 提取 x 分量
    local_idx[1] = (linear_idx >> voxel_to_block_depth_) & local_voxel_mask_;  // 提取 y 分量
    local_idx[2] = (linear_idx >> voxel_to_block_depth_double_) & local_voxel_mask_;  // 提取 z 分量
    
    // 计算全局体素索引
    Eigen::Vector3i voxel_idx;
    for (int i = 0; i < 3; i++) {
        voxel_idx[i] = (block_idx[i] << (block_depth_ - voxel_depth_)) + local_idx[i];
    }
    
    return voxel_idx;
}

// 线性索引转块索引
Eigen::Vector3i SOGMMap::linearToBlockIdx(int linear_idx) const {
    Eigen::Vector3i map_idx;
    map_idx[2] = linear_idx / block_num_xy_;
    linear_idx -= map_idx[2] * block_num_xy_;
    map_idx[1] = linear_idx / block_num_x_;
    map_idx[0] = linear_idx % block_num_x_;
    // linear_idx -= map_idx[1] * block_num_x_;
    // map_idx[0] = linear_idx;

    // map_idx[0] = linear_idx / block_num_yz_;  // 计算 x 分量
    // linear_idx -= map_idx[0] * block_num_yz_;  // 更新线性索引
    // map_idx[1] = linear_idx / block_num_z_;  // 计算 y 分量
    // map_idx[2] = linear_idx % block_num_z_;  // 计算 z 分量
    
    // 计算全局块索引
    Eigen::Vector3i block_idx;
    for (int i = 0; i < 3; i++) {
        map_idx[i] = (map_idx[i] - origin_block_[i]) % block_num_[i];
        if (map_idx[i] < 0) {
            map_idx[i] += block_num_[i];
        }
    }
    block_idx = map_idx + origin_block_;
    
    return block_idx;
}

// 检查子体素索引是否在地图范围内
bool SOGMMap::isSubVoxelIdxInMap(const Eigen::Vector3i& subvoxel_idx) const {
    Eigen::Vector3i block_idx;
    subVoxelIdxToBlockIdx(subvoxel_idx, block_idx);
    return isBlockIdxInMap(block_idx);
}

// 检查体素索引是否在地图范围内
bool SOGMMap::isVoxelIdxInMap(const Eigen::Vector3i& voxel_idx) const {
    Eigen::Vector3i block_idx;
    voxelIdxToBlockIdx(voxel_idx, block_idx);
    return isBlockIdxInMap(block_idx);
}

// 检查块索引是否在地图范围内
bool SOGMMap::isBlockIdxInMap(const Eigen::Vector3i& block_idx) const {
    Eigen::Vector3i rel_idx;
    for (int i = 0; i < 3; i++) {
        rel_idx[i] = block_idx[i] - origin_block_[i];
        // 考虑循环地图
        if (rel_idx[i] < 0 || rel_idx[i] >= block_num_[i]) {
            return false;
        }
    }
    return true;
}

void SOGMMap::getLocalMapState(std::vector<multilayer::VoxelGridMsg>& occupied_cells) const {
    occupied_cells.clear();
    // 预估大小，避免多次内存重分配
    occupied_cells.reserve(active_block_indices_.size() * 16); 

    // 不再遍历整个三维空间，而是只遍历被激活的块！
    // 效率从 O(N*M*K) 降低到 O(ActiveBlockCount)
    for (const int linear_idx : active_block_indices_) {
        if (linear_idx < 0 || linear_idx >= blocks_.size() || blocks_[linear_idx] == nullptr) {
            continue;
        }

        const Block* block = blocks_[linear_idx];

        // 虽然活动列表里的块理论上不应是free的，但为安全起见再检查一次
        if (block->is_free()) {
            continue; 
        }

        // 获取块的三维索引，用于后续计算
        Eigen::Vector3i current_block_idx = linearToBlockIdx(linear_idx);

        // 根据Block当前的层级，提取对应分辨率的占据信息
        switch (block->layer_) {
            case LayerType::BLOCK: {
                multilayer::VoxelGridMsg msg;
                Eigen::Vector3d pos;
                blockIdxToWorld(current_block_idx, pos);
                msg.position.x = pos.x();
                msg.position.y = pos.y();
                msg.position.z = pos.z();
                msg.layer = static_cast<uint8_t>(LayerType::BLOCK);
                occupied_cells.push_back(msg);
                break;
            }
            case LayerType::VOXEL: {
                if (block->is_voxel_allocated_) {
                    for (int i = 0; i < block->voxels_.size(); ++i) {
                        const Voxel* voxel = block->voxels_[i];
                        if (voxel != nullptr && !voxel->is_free_) {
                            multilayer::VoxelGridMsg msg;
                            Eigen::Vector3i voxel_idx = localLinearToVoxelIdx(i, current_block_idx);
                            Eigen::Vector3d pos;
                            voxelIdxToWorld(voxel_idx, pos);
                            msg.position.x = pos.x();
                            msg.position.y = pos.y();
                            msg.position.z = pos.z();
                            msg.layer = static_cast<uint8_t>(LayerType::VOXEL);
                            occupied_cells.push_back(msg);
                        }
                    }
                }
                break;
            }
            case LayerType::SUBVOXEL: {
                if (block->is_voxel_allocated_) {
                    for (int i = 0; i < block->voxels_.size(); ++i) {
                        const Voxel* voxel = block->voxels_[i];
                        if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
                            Eigen::Vector3i voxel_idx = localLinearToVoxelIdx(i, current_block_idx);
                            for (int j = 0; j < voxel->subvoxel_values_.size(); ++j) {
                                // 检查子体素是否被占据
                                if (voxel->subvoxel_values_[j] >= min_occupancy_log_) {
                                    multilayer::VoxelGridMsg msg;
                                    Eigen::Vector3i subvoxel_idx = localLinearToSubVoxelIdx(j, voxel_idx);
                                    Eigen::Vector3d pos;
                                    subVoxelIdxToWorld(subvoxel_idx, pos);
                                    msg.position.x = pos.x();
                                    msg.position.y = pos.y();
                                    msg.position.z = pos.z();
                                    msg.layer = static_cast<uint8_t>(LayerType::SUBVOXEL);
                                    occupied_cells.push_back(msg);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }
    }
}

// 添加到 SOGMMap 类的公有方法中
bool SOGMMap::saveSemanticKittiVoxels(const std::string& base_directory, 
                                      const Eigen::Vector3d& sensor_pos,
                                      double voxel_size,
                                      int grid_x, int grid_y, int grid_z) {
    // 确保目录存在
    if (!createDirectoryIfNotExists(base_directory)) {
        return false;
    }
    
    // 生成序列文件名
    std::ostringstream filename_stream;
    filename_stream << base_directory;
    if (base_directory.back() != '/') {
        filename_stream << '/';
    }
    filename_stream << std::setfill('0') << std::setw(6) << file_sequence_number << ".bin";
    std::string filename = filename_stream.str();
    
    // 增加序列号
    file_sequence_number++;

    size_t total_voxels = grid_x * grid_y * grid_z;
    std::vector<bool> occupied_grid;
    occupied_grid.reserve(total_voxels);
    
    // 计算网格的起始位置（以传感器为中心）
    Eigen::Vector3d grid_origin;
    grid_origin.x() = sensor_pos.x() - (grid_x * voxel_size) / 2.0;
    grid_origin.y() = sensor_pos.y() - (grid_y * voxel_size) / 2.0;
    grid_origin.z() = sensor_pos.z() - (grid_z * voxel_size) / 2.0;
    
    std::cout << "[SemanticKITTI] Saving voxel grid centered at sensor position: " 
              << sensor_pos.transpose() << std::endl;
    std::cout << "[SemanticKITTI] Grid origin: " << grid_origin.transpose() << std::endl;
    std::cout << "[SemanticKITTI] Grid size: " << grid_x << "x" << grid_y << "x" << grid_z 
              << ", voxel size: " << voxel_size << "m" << std::endl;
    
    int filled_voxels = 0;
    
    // 遍历网格中的每个体素
    for (int x = 0; x < grid_x; ++x) {
        for (int y = 0; y < grid_y; ++y) {
            for (int z = 0; z < grid_z; ++z) {
                // 计算当前体素的世界坐标
                Eigen::Vector3d voxel_world_pos;
                voxel_world_pos.x() = grid_origin.x() + (x + 0.5) * voxel_size;
                voxel_world_pos.y() = grid_origin.y() + (y + 0.5) * voxel_size;
                voxel_world_pos.z() = grid_origin.z() + (z + 0.5) * voxel_size;
                
                // 检查该位置是否被占据
                bool is_occupied = isOccupiedPrecise(voxel_world_pos);
                
                // 添加到布尔网格
                occupied_grid.push_back(is_occupied);
                
                if (is_occupied) {
                    filled_voxels++;
                }
            }
        }
    }
    
    std::cout << "[SemanticKITTI] Filled " << filled_voxels << " out of " 
              << (grid_x * grid_y * grid_z) << " voxels" << std::endl;
    
    // 打包成二进制位
    std::vector<unsigned char> binary_data;
    binary_data.reserve(occupied_grid.size() / 8 + 1);
    unsigned char current_byte = 0;
    
    for (size_t i = 0; i < occupied_grid.size(); ++i) {
        if (occupied_grid[i]) {
            current_byte |= (1 << (i % 8));
        }
        if ((i + 1) % 8 == 0 || i == occupied_grid.size() - 1) {
            binary_data.push_back(current_byte);
            current_byte = 0;
        }
    }
    
    // 写入二进制文件
    std::cout << "[SemanticKITTI] Writing to file: " << filename << std::endl;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[SemanticKITTI] ERROR: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // 写入二进制位数据
    file.write(reinterpret_cast<const char*>(binary_data.data()), binary_data.size());
    
    if (!file.good()) {
        std::cerr << "[SemanticKITTI] ERROR: Failed to write voxel data to " << filename << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    
    std::cout << "[SemanticKITTI] Successfully saved voxel grid to " << filename << std::endl;
    
    return true;
}


// 重载版本：使用默认参数
bool SOGMMap::saveSemanticKittiVoxels(const std::string& base_directory, 
                                      const Eigen::Vector3d& sensor_pos) {
    // 使用默认的 SemanticKITTI 参数
    double voxel_size = 0.2;  // 20cm 体素
    return saveSemanticKittiVoxels(base_directory, sensor_pos, voxel_size, 256, 256, 32);
}

// 修改创建目录的辅助函数，支持递归创建目录
bool SOGMMap::createDirectoryIfNotExists(const std::string& path) {
    // 检查路径是否为空
    if (path.empty()) {
        std::cerr << "[SemanticKITTI] ERROR: Empty path provided" << std::endl;
        return false;
    }
    
    struct stat info;
    
    // 如果目录已存在
    if (stat(path.c_str(), &info) == 0) {
        if (info.st_mode & S_IFDIR) {
            // 目录存在且是目录
            return true;
        } else {
            // 路径存在但不是目录
            std::cerr << "[SemanticKITTI] ERROR: " << path << " exists but is not a directory" << std::endl;
            return false;
        }
    }
    
    // 目录不存在，需要创建
    // 先尝试创建父目录
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        std::string parent_path = path.substr(0, pos);
        if (!createDirectoryIfNotExists(parent_path)) {
            return false;
        }
    }
    
    // 创建当前目录
    if (mkdir(path.c_str(), 0755) != 0) {
        int error = errno;
        std::cerr << "[SemanticKITTI] ERROR: Cannot create directory " << path 
                  << " - " << strerror(error) << " (errno: " << error << ")" << std::endl;
        
        // 提供一些常见错误的解决建议
        switch (error) {
            case EACCES:
                std::cerr << "[SemanticKITTI] HINT: Permission denied. Try running with sudo or check directory permissions." << std::endl;
                break;
            case ENOENT:
                std::cerr << "[SemanticKITTI] HINT: Parent directory does not exist." << std::endl;
                break;
            case EEXIST:
                // 如果目录已存在，这实际上是成功的
                if (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
                    std::cout << "[SemanticKITTI] Directory already exists: " << path << std::endl;
                    return true;
                }
                break;
            case ENOSPC:
                std::cerr << "[SemanticKITTI] HINT: No space left on device." << std::endl;
                break;
            default:
                std::cerr << "[SemanticKITTI] HINT: Unknown error occurred." << std::endl;
                break;
        }
        return false;
    }
    
    std::cout << "[SemanticKITTI] Successfully created directory: " << path << std::endl;
    return true;
}