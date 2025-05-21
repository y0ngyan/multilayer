#include "MultiLayerSOGMMap.hpp"
#include <set>

SOGMMap::SOGMMap()
{
}

SOGMMap::~SOGMMap()
{
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
    
    // 计算各级分辨率和单位数量
    subvoxel_num_in_voxel_ = (1 << voxel_depth_);  // 2^voxel_depth
    subvoxel_num_in_voxel_square_ = subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_;
    total_subvoxel_in_voxel_ = subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_ * subvoxel_num_in_voxel_;
    
    voxel_num_in_block_ = (1 << (block_depth_ - voxel_depth_));  // 2^(block_depth - voxel_depth)
    voxel_num_in_block_square_ = voxel_num_in_block_ * voxel_num_in_block_;
    total_voxel_in_block_ = voxel_num_in_block_ * voxel_num_in_block_ * voxel_num_in_block_;
    
    // 计算不同层级的分辨率
    voxel_res_ = sub_voxel_res_ * subvoxel_num_in_voxel_;
    block_res_ = voxel_res_ * voxel_num_in_block_;
    voxel_res_inv_ = 1.0 / voxel_res_;
    block_res_inv_ = 1.0 / block_res_;

    sub_voxel_radius_ = 0.5 * sub_voxel_res_ * sqrt(3.0);
    voxel_radius_ = 0.5 * voxel_res_ * sqrt(3.0);
    block_radius_ = 0.5 * block_res_ * sqrt(3.0);
    
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
    
    // 初始化块数组
    blocks_.resize(total_blocks, nullptr);
    for (int i = 0; i < total_blocks; ++i) {
        blocks_[i] = new Block(total_voxel_in_block_);
    }

    flag_traverse_ = std::vector<char>(total_blocks, 0);
    flag_rayend_ = std::vector<char>(total_blocks, 0);

    // 读取相机参数
    cv::FileNode DepthCamera_node = fs["DepthCamera"];
    depth_maxdist_ = (double)(DepthCamera_node["depth_maxdist"]);
    depth_mindist_ = (double)(DepthCamera_node["depth_mindist"]);
    skip_pixel_ = (int)(DepthCamera_node["skip_pixel"]);
    MIN_VALID_RATIO_ = (float)(DepthCamera_node["min_valid_ratio"]);
    depth_threshold_subvoxel_ = (double)(DepthCamera_node["depth_threshold_subvoxel"]);
    depth_threshold_voxel_ = (double)(DepthCamera_node["depth_threshold_voxel"]);
    depth_threshold_block_ = (double)(DepthCamera_node["depth_threshold_block_"]);

    // 读取相机内参
    depth_height_ = (int)(DepthCamera_node["height"]);
    depth_width_ = (int)(DepthCamera_node["width"]);
    fx_ = (double)(DepthCamera_node["fx"]);
    fy_ = (double)(DepthCamera_node["fy"]);
    cx_ = (double)(DepthCamera_node["cx"]);
    cy_ = (double)(DepthCamera_node["cy"]);
    k_depth_scaling_factor_ = (double)(DepthCamera_node["k_depth_scaling_factor"]);
    // 预计算深度缩放因子的倒数
    inv_depth_scaling_factor_ = 1.0 / k_depth_scaling_factor_;

    // 读取相机外参
    cv::Mat rc2b, tc2b;
    DepthCamera_node["R_C_2_B"] >> rc2b;
    DepthCamera_node["T_C_2_B"] >> tc2b;
    cv::cv2eigen(rc2b, R_C_2_B_);
    cv::cv2eigen(tc2b, T_C_2_B_);
    
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
    new_occ_.clear();
    new_free_.clear();

    // 初始化完成
    std::cout << "[SOGMMap] 初始化完成, 地图尺寸: " << map_x << " x " << map_y << " x " << map_z << " m" << std::endl;
    std::cout << "[SOGMMap] 子体素分辨率: " << sub_voxel_res_ << " m, 体素分辨率: " << voxel_res_ 
              << " m, 块分辨率: " << block_res_ << " m" << std::endl;
    std::cout << "[SOGMMap] 块数量: " << block_num_.transpose() << " 总块数: " << total_blocks << std::endl;
    
    fs.release();
}

void SOGMMap::update(pcl::PointCloud<pcl::PointXYZ> *ptws_hit_ptr, pcl::PointCloud<pcl::PointXYZ> *ptws_miss_ptr,
        const cv::Mat &depth_image, const Eigen::Matrix3d &R_C_2_W,
        const Eigen::Vector3d &T_C_2_W, Eigen::Vector3d camera_pos){
    slideMap(camera_pos);
    // ros::Time t1 = ros::Time::now();
    raycastProcess(ptws_hit_ptr, ptws_miss_ptr, camera_pos);
    voxelPolarProjectionProcessWithRaycast(depth_image, R_C_2_W, T_C_2_W);
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
    // 释放块中所有已分配的体素
    for (int i = 0; i < total_voxel_in_block_; i++) {
        block->free_voxel(i);
    }
    // 重置块的状态
    block->occupancy_value_ = 0.0f;
    block->is_free_ = true;
    block->layer_ = LayerType::BLOCK;
}

// main interface
std::vector<int> *SOGMMap::getSlideClearIndex()
{
    return &slideClearIndex_;
}

std::vector<int> *SOGMMap::getNewOcc()
{
    return &new_occ_;
}

std::vector<int> *SOGMMap::getNewFree()
{
    return &new_free_;
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
        cache.local_free_blocks.reserve(points_per_thread * 10); // 光线可能穿过多个空闲块
    }
    
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
                Eigen::Vector3d ray_start = pt_w;
                Eigen::Vector3d ray_end = camera_pos;
                
                // 光线步进参数化，使用块分辨率
                raycaster.setInput(ray_start, ray_end, block_res_);
                
                // 收集光线经过的所有块
                Eigen::Vector3i ray_block;
                while (raycaster.step(ray_block)) {
                    if (!isBlockIdxInMap(ray_block)) {
                        continue; // 跳过地图外的块
                    }
                    
                    int block_linear_idx;
                    blockIdxToLocalLinear(ray_block, block_linear_idx);
                    
                    // 标记未命中的块（光线穿过但不是终点）
                    bool already_marked = false;
                    #pragma omp atomic read
                    already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                    
                    if (already_marked) {
                        continue; // 如果已处理过，则跳过
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
                #pragma omp atomic read
                already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                
                if (!already_marked) {
                    #pragma omp atomic write
                    flag_traverse_[block_linear_idx] = current_raycast;
                    cache.processed_count++;
                    cache.local_free_blocks.push_back(block_linear_idx);
                }
                
                // 从未命中点到相机的光线投射
                Eigen::Vector3d ray_start = pt_w;
                Eigen::Vector3d ray_end = camera_pos;
                
                // 同样使用块分辨率进行光线投射
                raycaster.setInput(ray_start, ray_end, block_res_);
                
                // 处理光线经过的所有块
                Eigen::Vector3i ray_block;
                while (raycaster.step(ray_block)) {
                    if (!isBlockIdxInMap(ray_block)) {
                        continue; // 跳过地图外的块
                    }
                    
                    blockIdxToLocalLinear(ray_block, block_linear_idx);
                    
                    bool already_marked = false;
                    #pragma omp atomic read
                    already_marked = (flag_traverse_[block_linear_idx] == current_raycast);
                    
                    if (already_marked) {
                        continue; // 如果已处理过，则跳过
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

bool SOGMMap::isInDepthImage(const cv::Mat &depth_image, 
        const Eigen::Matrix3d &R_W_2_C,
        const Eigen::Vector3d &T_W_2_C, 
        const Eigen::Vector3d &voxel_center,
        double radius){
    // 计算体素在相机坐标系下的位置
    Eigen::Vector3d voxel_camera = R_W_2_C * voxel_center + T_W_2_C;
    
    // 忽略相机后方的体素
    if (voxel_camera.z() <= 0) return false;
    
    double distance = voxel_camera.norm();
    double min_azimuth, max_azimuth, min_elevation, max_elevation;
    
    // 计算体素的极坐标范围
    computeAngularBounds(voxel_camera, distance, radius, min_azimuth, max_azimuth, min_elevation, max_elevation);
    
    // 转换为像素坐标范围
    int min_u, min_v, max_u, max_v;
    polarToPixel(min_azimuth, min_elevation, min_u, min_v);
    polarToPixel(max_azimuth, max_elevation, max_u, max_v);
    
    // 确保值有效且在图像范围内
    min_u = std::max(0, std::min(min_u, depth_width_ - 1));
    max_u = std::max(0, std::min(max_u, depth_width_ - 1));
    min_v = std::max(0, std::min(min_v, depth_height_ - 1));
    max_v = std::max(0, std::min(max_v, depth_height_ - 1));
    
    // 确保min <= max
    if (min_u > max_u || min_v > max_v) {
        return false;  // 无效范围
    }
    
    // 如果范围太小，跳过
    if (max_u - min_u < 1 || max_v - min_v < 1) {
        return false;
    }

    return true;
}

// 新增：体素投影到深度图的辅助函数
bool SOGMMap::projectVoxelToDepthImage(const cv::Mat &depth_image, 
                                      const Eigen::Matrix3d &R_W_2_C,
                                      const Eigen::Vector3d &T_W_2_C, 
                                      const Eigen::Vector3d &voxel_center,
                                      double radius,
                                      double depth_threshold) {
    // 计算体素在相机坐标系下的位置
    Eigen::Vector3d voxel_camera = R_W_2_C * voxel_center + T_W_2_C;
    
    // 忽略相机后方的体素
    if (voxel_camera.z() <= 0) return false;
    
    double distance = voxel_camera.norm();
    double min_azimuth, max_azimuth, min_elevation, max_elevation;
    
    // 计算体素的极坐标范围
    computeAngularBounds(voxel_camera, distance, radius, min_azimuth, max_azimuth, min_elevation, max_elevation);
    
    // 转换为像素坐标范围
    int min_u, min_v, max_u, max_v;
    polarToPixel(min_azimuth, min_elevation, min_u, min_v);
    polarToPixel(max_azimuth, max_elevation, max_u, max_v);
    
    // 确保值有效且在图像范围内
    min_u = std::max(0, std::min(min_u, depth_width_ - 1));
    max_u = std::max(0, std::min(max_u, depth_width_ - 1));
    min_v = std::max(0, std::min(min_v, depth_height_ - 1));
    max_v = std::max(0, std::min(max_v, depth_height_ - 1));
    
    // 确保min <= max
    if (min_u > max_u || min_v > max_v) {
        return false;  // 无效范围
    }
    
    // 如果范围太小，跳过
    if (max_u - min_u < 1 || max_v - min_v < 1) {
        return false;
    }
    
    // 计算范围内的平均深度
    double avg_pixel_depth = 0;
    if (!getAverageDepth(depth_image, min_u, max_u, min_v, max_v, avg_pixel_depth)) {
        return false;  // 无效深度
    }
    
    // 检查体素深度与图像深度是否接近
    return fabs(voxel_camera.z() - avg_pixel_depth) < depth_threshold;
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

void SOGMMap::switchLayer(int block_idx, const Eigen::Vector3d& sensor_pos) {
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
    if (block->layer_ == target_layer) {
        return;
    }
    
    // 存储当前占据值，用于向下传递
    float current_occupancy = block->occupancy_value_;
    bool is_occupied = !block->is_free_;
    
    // 根据目标层级切换表示
    switch (target_layer) {
        case LayerType::BLOCK: {
            // 从更精细的表示转为块级别
            if (block->is_voxel_allocated_) {
                // 先向上传递，更新块的占据状态
                propagateOccupancyUp(block);
                
                // 释放所有体素内存
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    block->free_voxel(i);
                }
            }
            
            // 设置层级
            block->layer_ = LayerType::BLOCK;
            break;
        }
        
        case LayerType::VOXEL: {
            if (block->layer_ == LayerType::BLOCK) {
                // 从块级别升级到体素级别
                // 分配体素并将块的占据信息向下传递
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                propagateOccupancyDown(block, current_occupancy * 0.6f);
            } 
            else if (block->layer_ == LayerType::SUBVOXEL) {
                // 从子体素级别降级到体素级别
                // 对每个体素，聚合其子体素的占据信息
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    if (block->voxels_[i] != nullptr && block->voxels_[i]->is_subvoxel_allocated_) {
                        // 计算子体素的平均占据值
                        float subvoxel_sum = 0.0f;
                        int occupied_subvoxels = 0;
                        
                        for (const auto& subvoxel_value : block->voxels_[i]->subvoxel_values_) {
                            if (subvoxel_value > min_occupancy_log_) {
                                subvoxel_sum += subvoxel_value;
                                occupied_subvoxels++;
                            }
                        }
                        
                        // 更新体素的占据信息
                        if (occupied_subvoxels > 0) {
                            block->voxels_[i]->occupancy_value_ = subvoxel_sum / occupied_subvoxels;
                            block->voxels_[i]->is_free_ = false;
                        } else {
                            block->voxels_[i]->occupancy_value_ = 0.0f;
                            block->voxels_[i]->is_free_ = true;
                        }
                        
                        // 释放子体素内存
                        block->voxels_[i]->free_subvoxels();
                    }
                }
                
                // 向上传递更新块的状态
                propagateOccupancyUp(block);
            }
            
            // 设置层级
            block->layer_ = LayerType::VOXEL;
            break;
        }
        
        case LayerType::SUBVOXEL: {
            if (!block->is_voxel_allocated_) {
                // 如果还没有体素，先分配体素
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                // 向下传递块的占据信息到体素
                propagateOccupancyDown(block, current_occupancy * 0.6f);
            }
            
            // 为每个体素分配子体素
            for (int i = 0; i < block->voxels_.size(); ++i) {
                Voxel* voxel = block->voxels_[i];
                if (voxel != nullptr && !voxel->is_subvoxel_allocated_) {
                    voxel->allocate_subvoxels(total_subvoxel_in_voxel_, voxel->occupancy_value_);
                }
            }
            
            // 设置层级并更新块状态
            block->layer_ = LayerType::SUBVOXEL;
            propagateOccupancyUp(block);
            break;
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
    if (block->layer_ == target_layer) {
        return;
    }
    
    // 存储当前占据值，用于向下传递
    float current_occupancy = block->occupancy_value_;
    bool is_occupied = !block->is_free_;
    bool use_depth_check = !depth_image.empty();
    
    // 根据目标层级切换表示
    switch (target_layer) {
        case LayerType::BLOCK: {
            // 从更精细的表示转为块级别
            if (block->is_voxel_allocated_) {
                // 释放所有体素内存
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    block->free_voxel(i);
                }
            }
            
            // 设置层级
            block->layer_ = LayerType::BLOCK;
            break;
        }
        
        case LayerType::VOXEL: {
            if (block->layer_ == LayerType::BLOCK) {
                // 从块级别升级到体素级别
                // 分配体素并将块的占据信息向下传递
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                
                // 检查深度图投影
                if (use_depth_check) {
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel == nullptr) continue;
                        
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                        Eigen::Vector3d voxel_center;
                        voxelIdxToWorld(voxel_grid_idx, voxel_center);
                        
                        // 块是占用的，只有占据体素继承块占据概率
                        if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                            voxel_center, voxel_radius_)) {
                            voxel->occupancy_value_ = current_occupancy;
                            voxel->is_free_ = block->is_free();
                        } else {
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }
                } else {
                    // 没有深度图，直接继承父级概率
                    propagateOccupancyDown(block, current_occupancy);
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
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                
                // 检查深度图投影
                if (use_depth_check) {
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel == nullptr) continue;
                        
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                        Eigen::Vector3d voxel_center;
                        voxelIdxToWorld(voxel_grid_idx, voxel_center);
                        // 块是占用的，只有占据体素继承块占据概率
                        if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                            voxel_center, voxel_radius_)) {
                            voxel->occupancy_value_ = current_occupancy;
                            voxel->is_free_ = block->is_free();
                        } else {
                            voxel->occupancy_value_ = 0.0f;
                            voxel->is_free_ = true;
                        }
                    }
                } else {
                    // 没有深度图，直接继承父级概率
                    propagateOccupancyDown(block, current_occupancy);
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
                    
                    // 检查子体素的深度投影
                    if (use_depth_check) {
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(i, block_grid_idx);
                        
                        for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                            Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                            Eigen::Vector3d subvoxel_center;
                            subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

                            if (voxel_is_free) {
                                // 块是空闲的，只有空闲子体素才继承块占据概率
                                if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                    subvoxel_center, sub_voxel_radius_)) {
                                    voxel->subvoxel_values_[subvox_idx] = current_occupancy;
                                } else {
                                    voxel->subvoxel_values_[subvox_idx] = 0.0f;
                                }
                            }
                            else{
                                voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                            }
                        }
                    } else {
                        // 没有深度图，所有子体素继承体素的概率
                        for (int j = 0; j < voxel->subvoxel_values_.size(); ++j) {
                            voxel->subvoxel_values_[j] = voxel_occupancy;
                        }
                    }
                }
            }
            // 设置层级并更新块状态
            block->layer_ = LayerType::SUBVOXEL;
            // propagateOccupancyUp(block);
            break;
        }
    }
}

// 修改switchLayerWithProject函数签名，添加用于收集层级变化信息的参数
void SOGMMap::switchLayerWithProjectWithUpdateGlobal(int block_idx, const Eigen::Vector3d& sensor_pos, 
                          const cv::Mat& depth_image,
                          const Eigen::Matrix3d& R_W_2_C,
                          const Eigen::Vector3d& T_W_2_C,
                          std::vector<LayerVoxel>& layer_change_freed,
                          std::vector<LayerVoxel>& layer_change_occupied) {
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
    if (block->layer_ == target_layer) {
        return;
    }
    
    // 存储当前占据值和层级信息，用于记录变化
    float current_occupancy = block->occupancy_value_;
    bool is_occupied = !block->is_free_;
    bool use_depth_check = !depth_image.empty();
    LayerType old_layer = block->layer_;
    
    // 如果当前块是占据的，先记录旧的层级信息用于释放
    if (is_occupied) {
        // 记录释放的旧层级信息
        if (old_layer == LayerType::BLOCK) {
            // 整个块将被替换
            layer_change_freed.emplace_back(block_center, block_res_, 
                                           current_occupancy, LayerType::BLOCK);
        }
        else if (old_layer == LayerType::VOXEL && block->is_voxel_allocated_) {
            // 记录所有非空闲的体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || voxel->is_free_) continue;
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                Eigen::Vector3d voxel_center;
                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                
                layer_change_freed.emplace_back(voxel_center, voxel_res_, 
                                               voxel->occupancy_value_, LayerType::VOXEL);
            }
        }
        else if (old_layer == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
            // 记录所有占据的子体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || !voxel->is_subvoxel_allocated_) continue;
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                
                for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                    if (voxel->subvoxel_values_[subvox_idx] >= min_occupancy_log_) {
                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                        Eigen::Vector3d subvoxel_center;
                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                        
                        layer_change_freed.emplace_back(subvoxel_center, sub_voxel_res_,
                                                       voxel->subvoxel_values_[subvox_idx], 
                                                       LayerType::SUBVOXEL);
                    }
                }
            }
        }
    }
    
    // 根据目标层级切换表示
    switch (target_layer) {
        case LayerType::BLOCK: {
            // 从更精细的表示转为块级别
            if (block->is_voxel_allocated_) {
                // 先向上传递，更新块的占据状态
                // propagateOccupancyUp(block);
                
                // 释放所有体素内存
                for (int i = 0; i < block->voxels_.size(); ++i) {
                    block->free_voxel(i);
                }
            }
            
            // 设置层级
            block->layer_ = LayerType::BLOCK;
            break;
        }
        
        case LayerType::VOXEL: {
            if (block->layer_ == LayerType::BLOCK) {
                // 从块级别升级到体素级别
                // 分配体素并将块的占据信息向下传递
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                
                // 检查深度图投影
                if (use_depth_check) {
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel == nullptr) continue;
                        
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                        Eigen::Vector3d voxel_center;
                        voxelIdxToWorld(voxel_grid_idx, voxel_center);

                        if (block->is_free()) {
                            // 块是空闲的，只有空闲体素才继承块占据概率
                            if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                       voxel_center, voxel_radius_)) {
                                voxel->occupancy_value_ = current_occupancy;
                                voxel->is_free_ = block->is_free();
                            } else {
                                voxel->occupancy_value_ = 0.0f;
                                voxel->is_free_ = true;
                            }
                        }
                        else{
                            // 块是占用的，只有占据体素继承块占据概率
                            if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                            voxel_center, voxel_radius_)) {
                                voxel->occupancy_value_ = current_occupancy;
                                voxel->is_free_ = block->is_free();
                            } else {
                                if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                       voxel_center, voxel_radius_, depth_threshold_voxel_)) {
                                    voxel->occupancy_value_ = current_occupancy;
                                    voxel->is_free_ = block->is_free();
                                } else {
                                voxel->occupancy_value_ = 0.0f;
                                voxel->is_free_ = true;
                                }
                            }
                        }            
                    }
                } else {
                    // 没有深度图，直接继承父级概率
                    propagateOccupancyDown(block, current_occupancy);
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
                
                // 向上传递更新块的状态
                // propagateOccupancyUp(block);
            }
            
            // 设置层级
            block->layer_ = LayerType::VOXEL;
            break;
        }
        
        case LayerType::SUBVOXEL: {
            if (!block->is_voxel_allocated_) {
                // 如果还没有体素，先分配体素
                block->allocate_voxels(total_voxel_in_block_, 0.0f);
                
                // 检查深度图投影
                if (use_depth_check) {
                    for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                        Voxel* voxel = block->voxels_[vox_idx];
                        if (voxel == nullptr) continue;
                        
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                        Eigen::Vector3d voxel_center;
                        voxelIdxToWorld(voxel_grid_idx, voxel_center);

                        if (block->is_free()) {
                            // 块是空闲的，只有空闲体素才继承块占据概率
                            if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                       voxel_center, voxel_radius_)) {
                                voxel->occupancy_value_ = current_occupancy;
                                voxel->is_free_ = block->is_free();
                            } else {
                                voxel->occupancy_value_ = 0.0f;
                                voxel->is_free_ = true;
                            }
                        }
                        else{
                            // 块是占用的，只有占据体素继承块占据概率
                            if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                            voxel_center, voxel_radius_)) {
                                voxel->occupancy_value_ = current_occupancy;
                                voxel->is_free_ = block->is_free();
                            } else {
                                if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                       voxel_center, voxel_radius_, depth_threshold_voxel_)) {
                                    voxel->occupancy_value_ = current_occupancy;
                                    voxel->is_free_ = block->is_free();
                                } else {
                                voxel->occupancy_value_ = 0.0f;
                                voxel->is_free_ = true;
                                }
                            }
                        } 
                    }
                } else {
                    // 没有深度图，直接继承父级概率
                    propagateOccupancyDown(block, current_occupancy);
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
                    
                    // 检查子体素的深度投影
                    if (use_depth_check) {
                        // 获取体素的世界坐标
                        Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(i, block_grid_idx);
                        
                        for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                            Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                            Eigen::Vector3d subvoxel_center;
                            subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

                            if (voxel_is_free) {
                                // 块是空闲的，只有空闲子体素才继承块占据概率
                                if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                    subvoxel_center, sub_voxel_radius_)) {
                                    voxel->subvoxel_values_[subvox_idx] = current_occupancy;
                                    // voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                                } else {
                                    voxel->subvoxel_values_[subvox_idx] = 0.0f;
                                }
                            }
                            else{
                                // 块是占用的，只有占据子体素继承块占据概率
                                if (!isInDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                            subvoxel_center, sub_voxel_radius_)) {
                                    voxel->subvoxel_values_[subvox_idx] = voxel_occupancy;
                                } else {
                                    voxel->subvoxel_values_[subvox_idx] = 0.0f;
                                }
                            }
                        }
                    } else {
                        // 没有深度图，所有子体素继承体素的概率
                        for (int j = 0; j < voxel->subvoxel_values_.size(); ++j) {
                            voxel->subvoxel_values_[j] = voxel_occupancy;
                        }
                    }
                }
            }
            // 设置层级并更新块状态
            block->layer_ = LayerType::SUBVOXEL;
            break;
        }
    }
    
    // 现在收集新层级的占据信息（如果块被占据）
    if (!block->is_free_) {
        // 根据目标层级收集占据信息
        if (target_layer == LayerType::BLOCK) {
            layer_change_occupied.emplace_back(block_center, block_res_, 
                                              block->occupancy_value_, LayerType::BLOCK);
        }
        else if (target_layer == LayerType::VOXEL && block->is_voxel_allocated_) {
            // 收集所有非空闲的体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || voxel->is_free_) continue;
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                Eigen::Vector3d voxel_center;
                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                
                layer_change_occupied.emplace_back(voxel_center, voxel_res_, 
                                                 voxel->occupancy_value_, LayerType::VOXEL);
            }
        }
        else if (target_layer == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
            // 收集所有占据的子体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || !voxel->is_subvoxel_allocated_ || voxel->is_free_) continue;
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                
                for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                    if (voxel->subvoxel_values_[subvox_idx] >= min_occupancy_log_) {
                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                        Eigen::Vector3d subvoxel_center;
                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                        
                        layer_change_occupied.emplace_back(subvoxel_center, sub_voxel_res_,
                                                         voxel->subvoxel_values_[subvox_idx], 
                                                         LayerType::SUBVOXEL);
                    }
                }
            }
        }
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

bool SOGMMap::getAverageDepth(const cv::Mat &depth_image, 
                                int min_u, int max_u,
                                int min_v, int max_v,
                                double &avg_depth) {
    // 计算范围内像素的平均深度
    double sum_depth = 0.0;
    int valid_pixels = 0;
    int total_sampled_pixels = 0;  // 新增：记录采样的总像素数
    int skip = std::max(1, (max_u - min_u) / 10); // 动态计算采样步长
    for (int v = min_v; v <= max_v; v += skip) {
        for (int u = min_u; u <= max_u; u += skip) {
            total_sampled_pixels++;  // 增加采样像素计数
            double pixel_depth = depth_image.at<uint16_t>(v, u) * inv_depth_scaling_factor_;
            
            // 只考虑有效深度值
            if (pixel_depth > depth_mindist_ && pixel_depth <= depth_maxdist_) {
                sum_depth += pixel_depth;
                valid_pixels++;
            }
        }
    }
    float valid_ratio = static_cast<float>(valid_pixels) / total_sampled_pixels;
    if (valid_pixels == 0 || valid_ratio < MIN_VALID_RATIO_) {
        return false;
    }
    // 计算平均深度
    avg_depth = sum_depth / valid_pixels;
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

// 线性索引转换
void SOGMMap::subVoxelIdxToLocalLinear(const Eigen::Vector3i& subvoxel_idx, int& local_linear_idx) const{
    // 获取子体素在体素内的局部索引
    Eigen::Vector3i voxel_idx;
    subVoxelIdxToVoxelIdx(subvoxel_idx, voxel_idx);
    
    Eigen::Vector3i local_idx;
    for (int i = 0; i < 3; i++) {
        local_idx[i] = subvoxel_idx[i] - (voxel_idx[i] << voxel_depth_);  // 获取余数
    }
    
    // 转换为线性索引
    local_linear_idx = local_idx[0] + 
                       local_idx[1] * subvoxel_num_in_voxel_ + 
                       local_idx[2] * subvoxel_num_in_voxel_square_;
}
void SOGMMap::voxelIdxToLocalLinear(const Eigen::Vector3i& voxel_idx, int& local_linear_idx) const{
    // 获取体素在块内的局部索引
    Eigen::Vector3i block_idx;
    voxelIdxToBlockIdx(voxel_idx, block_idx);
    
    Eigen::Vector3i local_idx;
    for (int i = 0; i < 3; i++) {
        local_idx[i] = voxel_idx[i] - (block_idx[i] << (block_depth_ - voxel_depth_));  // 获取余数
    }
    
    // 转换为线性索引
    local_linear_idx = local_idx[0] + 
                       local_idx[1] * voxel_num_in_block_ + 
                       local_idx[2] * voxel_num_in_block_square_;
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
}

// 局部线性索引转子体素索引
Eigen::Vector3i SOGMMap::localLinearToSubVoxelIdx(int linear_idx, const Eigen::Vector3i& voxel_idx) const {
    Eigen::Vector3i local_idx;
    local_idx[2] = linear_idx % subvoxel_num_in_voxel_;
    linear_idx /= subvoxel_num_in_voxel_;
    local_idx[1] = linear_idx % subvoxel_num_in_voxel_;
    local_idx[0] = linear_idx / subvoxel_num_in_voxel_;
    
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
    local_idx[2] = linear_idx % voxel_num_in_block_;
    linear_idx /= voxel_num_in_block_;
    local_idx[1] = linear_idx % voxel_num_in_block_;
    local_idx[0] = linear_idx / voxel_num_in_block_;
    
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
    linear_idx -= map_idx[1] * block_num_x_;
    map_idx[0] = linear_idx;

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

void SOGMMap::voxelPolarProjectionProcessWithRaycast(const cv::Mat &depth_image, 
                                                  const Eigen::Matrix3d &R_C_2_W, 
                                                  const Eigen::Vector3d &T_C_2_W) {
    Eigen::Matrix3d R_W_2_C = R_C_2_W.transpose();
    Eigen::Vector3d T_W_2_C = -R_W_2_C * T_C_2_W;

    new_occ_.clear();
    new_free_.clear();

    // 多线程处理的互斥锁，用于保护共享资源
    std::mutex new_free_mutex;
    std::mutex new_occ_mutex;

    // 为多线程处理准备线程本地变量容器
    std::vector<std::vector<int>> thread_new_free(num_projection_threads_);
    std::vector<std::vector<int>> thread_new_occ(num_projection_threads_);

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
        
        free_threads.emplace_back([this, &thread_new_free, &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int block_idx = free_blocks_[i];
                if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
                    continue;
                }
                
                Block* block = blocks_[block_idx];

                // 记录更新前的状态
                bool was_occupied = !block->is_free_;
                uint8_t free_voxels=0;
                uint8_t free_subvoxels=0;

                // 获取块的世界坐标
                Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
                Eigen::Vector3d block_center;
                blockIdxToWorld(block_grid_idx, block_center);
                
                // 计算块到相机的距离
                Eigen::Vector3d block_camera = R_W_2_C * block_center + T_W_2_C;
                double distance = block_camera.norm();
                
                // 根据距离调整层级 - 这里需要加锁
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);

                    if (block->is_free()){
                        if (block->is_voxel_allocated_) {
                            for (int i = 0; i < block->voxels_.size(); ++i) {
                                block->free_voxel(i);
                            }
                        }
                        block->layer_ = LayerType::BLOCK;
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
                        continue;
                    }

                    // switchLayer(block_idx, T_C_2_W);
                    switchLayerWithProject(block_idx, T_C_2_W, depth_image, R_W_2_C, T_W_2_C);
                    
                    // 重新获取块引用（因为switchLayer可能已修改了块）
                    block = blocks_[block_idx];
                    
                    // 根据层级更新占据概率
                    if (block->layer_ == LayerType::BLOCK) {
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
                    } 
                    else if (block->layer_ == LayerType::VOXEL) {
                        // 对每个体素更新概率
                        if (block->is_voxel_allocated_) {
                            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                                Voxel* voxel = block->voxels_[vox_idx];
                                if (voxel != nullptr) {
                                    // 获取体素的世界坐标
                                    Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                                    Eigen::Vector3d voxel_center;
                                    voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                    // 检查体素是否在深度图像中
                                    if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, voxel_center, voxel_radius_)) {
                                        updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_miss_log_);
                                    }
                                    if (voxel->is_free_) {
                                        free_voxels++;
                                    }
                                }
                            }
                            // 向上传递更新后的占据信息
                            propagateOccupancyUp(block);
                        }
                    }
                    else if (block->layer_ == LayerType::SUBVOXEL) {
                        // 对子体素进行更新
                        if (block->is_voxel_allocated_) {
                            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                                Voxel* voxel = block->voxels_[vox_idx];
                                if (voxel == nullptr && !(voxel->is_subvoxel_allocated_)) continue;
                                // 获取体素的世界坐标
                                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                                Eigen::Vector3d voxel_center;
                                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, voxel_center, voxel_radius_)) {
                                    for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                                        // 获取子体素的世界坐标
                                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                                        Eigen::Vector3d subvoxel_center;
                                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                                        // 检查子体素是否在深度图像中
                                        if (isInDepthImage(depth_image, R_W_2_C, T_W_2_C, subvoxel_center, sub_voxel_radius_)) {
                                            float &value = voxel->subvoxel_values_[subvox_idx];
                                            float new_value = value + prob_miss_log_;
                                            value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                            if (value < min_occupancy_log_) {
                                                free_subvoxels++;
                                            }
                                        }
                                    }
                                }
                            }
                            // 向上传递更新后的占据信息
                            propagateOccupancyUp(block);
                        }
                    }
                    
                    // 检查状态是否从占据变为空闲
                    // if ((was_occupied && block->is_free_) ||
                    //     (free_voxels >= total_voxel_in_block_ /4) ||
                    //     (free_subvoxels >= total_subvoxel_in_voxel_ /4)) {
                    //     thread_new_free[t].push_back(block_idx);
                    // }
                    if ((was_occupied && block->is_free_)||
                        (was_occupied && free_voxels >= total_voxel_in_block_ / 4) ||
                        (was_occupied && free_subvoxels >= total_subvoxel_in_voxel_ / 4)) {
                        thread_new_free[t].push_back(block_idx);
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
    
    // 2. 处理占据块部分 - 多线程
    int occ_count = occ_blocks_.size();
    blocks_per_thread = (occ_count + num_projection_threads_ - 1) / num_projection_threads_;
    
    std::vector<std::thread> occ_threads;
    occ_threads.reserve(num_projection_threads_);

    // 用于存储各线程的层级变化信息
    std::vector<std::vector<LayerVoxel>> thread_layer_change_freed(num_projection_threads_);
    std::vector<std::vector<LayerVoxel>> thread_layer_change_occupied(num_projection_threads_);
    
    for (int t = 0; t < num_projection_threads_; ++t) {
        size_t start_idx = t * blocks_per_thread;
        size_t end_idx = std::min(start_idx + blocks_per_thread, static_cast<size_t>(occ_count));
        
        // 如果这个线程没有要处理的块，则跳过
        if (start_idx >= end_idx) {
            continue;
        }
        
        occ_threads.emplace_back([this, &thread_new_occ, &thread_layer_change_freed, &thread_layer_change_occupied,
            &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                int block_idx = occ_blocks_[i];
                if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
                    continue;
                }
                
                // 加锁保护块访问
                {
                    std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
                    Block* block = blocks_[block_idx];

                    // 记录更新前的状态
                    bool was_free = block->is_free_;
                    
                    // 获取块的世界坐标
                    Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
                    Eigen::Vector3d block_center;
                    blockIdxToWorld(block_grid_idx, block_center);

                    // switchLayerWithProject(block_idx, T_C_2_W, depth_image, R_W_2_C, T_W_2_C);
                    std::vector<LayerVoxel> block_layer_freed;
                    std::vector<LayerVoxel> block_layer_occupied;
                    switchLayerWithProjectWithUpdateGlobal(block_idx, T_C_2_W, depth_image, R_W_2_C, T_W_2_C, block_layer_freed, block_layer_occupied);
                    // 将该块的层级变化信息添加到线程本地存储
                    thread_layer_change_freed[t].insert(thread_layer_change_freed[t].end(),
                                                  block_layer_freed.begin(), block_layer_freed.end());
                    thread_layer_change_occupied[t].insert(thread_layer_change_occupied[t].end(),
                                                    block_layer_occupied.begin(), block_layer_occupied.end());

                    // 根据距离调整层级
                    // switchLayer(block_idx, T_C_2_W);
                    
                    // 重新获取块引用（因为switchLayer可能已修改了块）
                    block = blocks_[block_idx];

                    // 根据层级进行处理
                    bool block_updated = false;
                    
                    if (block->layer_ == LayerType::BLOCK) {
                        // Block层级：直接增加占据概率
                        updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
                        block_updated = true;
                    } 
                    else if (block->layer_ == LayerType::VOXEL) {
                        // Voxel层级：将块中所有voxel投影到深度图中
                        if (block->is_voxel_allocated_) {
                            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                                Voxel* voxel = block->voxels_[vox_idx];
                                if (voxel == nullptr) continue;
                                
                                // 获取体素的世界坐标
                                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                                Eigen::Vector3d voxel_center;
                                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                
                                // 投影体素到深度图并检查是否匹配
                                if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                            voxel_center, voxel_radius_, depth_threshold_voxel_)) {
                                    // 更新体素的占据概率
                                    updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
                                    block_updated = true;
                                }
                            }
                            
                            // 如果任何体素被更新，向上传递更新块的状态
                            if (block_updated) {
                                propagateOccupancyUp(block);
                            }
                        }
                    } 
                    else if (block->layer_ == LayerType::SUBVOXEL) {
                        // Subvoxel层级：检查体素级别的投影，然后更新子体素
                        if (block->is_voxel_allocated_) {
                            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                                Voxel* voxel = block->voxels_[vox_idx];
                                if (voxel == nullptr || !voxel->is_subvoxel_allocated_) {
                                    continue;
                                }
                                
                                // 获取体素的世界坐标
                                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                                Eigen::Vector3d voxel_center;
                                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                
                                // 投影体素到深度图并检查是否匹配
                                if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                           voxel_center, voxel_radius_, depth_threshold_voxel_)) {
                                    // 处理子体素级别的投影
                                    bool any_subvoxel_updated = false;
                                    // 遍历体素内的所有子体素
                                    for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                                        if (voxel->is_subvoxel_allocated_) {
                                            Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                                            Eigen::Vector3d subvoxel_center;
                                            subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

                                            // 投影子体素到深度图并检查是否匹配
                                            if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
                                                                     subvoxel_center, sub_voxel_radius_, depth_threshold_subvoxel_)) {
                                                float &value = voxel->subvoxel_values_[subvox_idx];
                                                float new_value = value + prob_hit_log_;
                                                value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
                                                any_subvoxel_updated = true;
                                            }
                                        }
                                    }
                                    // 如果有任何子体素被更新，向上传递更新
                                    if (any_subvoxel_updated) {
                                        block_updated = true;
                                    }
                                }
                            }
                            
                            // 如果任何体素被更新，向上传递更新块的状态
                            if (block_updated) {
                                propagateOccupancyUp(block);
                            }
                        }
                    }
                    
                    // 检查状态是否从空闲变为占据
                    if (was_free && !block->is_free_) {
                        thread_new_occ[t].push_back(block_idx);
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
    
    // 合并所有线程结果到主结果容器
    for (int t = 0; t < num_projection_threads_; ++t) {
        new_free_.insert(new_free_.end(), thread_new_free[t].begin(), thread_new_free[t].end());
        new_occ_.insert(new_occ_.end(), thread_new_occ[t].begin(), thread_new_occ[t].end());
    }
    
    // 收集多层级的体素
    collectMultiLayerVoxels();

    // 收集多层级信息前，先收集层级变化信息
    std::vector<LayerVoxel> all_layer_freed;
    std::vector<LayerVoxel> all_layer_occupied;

    for (int t = 0; t < num_projection_threads_; ++t) {
        all_layer_freed.insert(all_layer_freed.end(), 
                             thread_layer_change_freed[t].begin(), 
                             thread_layer_change_freed[t].end());
        all_layer_occupied.insert(all_layer_occupied.end(), 
                               thread_layer_change_occupied[t].begin(), 
                               thread_layer_change_occupied[t].end());
    }

    // 将层级变化收集的体素添加到返回结果中
    new_freed_voxels_.insert(new_freed_voxels_.end(), all_layer_freed.begin(), all_layer_freed.end());
    new_occupied_voxels_.insert(new_occupied_voxels_.end(), all_layer_occupied.begin(), all_layer_occupied.end());

    // 清空处理过的块列表
    free_blocks_.clear();
    occ_blocks_.clear();
}

// 收集多层级体素的函数实现
void SOGMMap::collectMultiLayerVoxels() {
    // 清空旧数据
    new_occupied_voxels_.clear();
    new_freed_voxels_.clear();
    
    // 预留空间
    new_occupied_voxels_.reserve(new_occ_.size() * 8);  // 估计值，体素比块多
    new_freed_voxels_.reserve(new_free_.size() * 8);
    
    // 处理新占据的块
    for (const auto& block_idx : new_occ_) {
        if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
            continue;
        }
        
        Block* block = blocks_[block_idx];
        Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
        Eigen::Vector3d block_center;
        blockIdxToWorld(block_grid_idx, block_center);
        
        // 根据层级添加不同分辨率的体素
        if (block->layer_ == LayerType::BLOCK) {
            // 整个块作为单个单元
            new_occupied_voxels_.emplace_back(block_center, block_res_, 
                                             block->occupancy_value_, LayerType::BLOCK);
        }
        else if (block->layer_ == LayerType::VOXEL && block->is_voxel_allocated_) {
            // 添加块中的占据体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || voxel->is_free_) {
                    continue;
                }
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                Eigen::Vector3d voxel_center;
                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                
                new_occupied_voxels_.emplace_back(voxel_center, voxel_res_, 
                                                 voxel->occupancy_value_, LayerType::VOXEL);
            }
        }
        else if (block->layer_ == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
            // 添加块中的占据子体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || !voxel->is_subvoxel_allocated_) {
                    continue;
                }
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                
                for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                    if (voxel->subvoxel_values_[subvox_idx] >= min_occupancy_log_) {
                        // 子体素被占据
                        Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                        Eigen::Vector3d subvoxel_center;
                        subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                        
                        new_occupied_voxels_.emplace_back(subvoxel_center, sub_voxel_res_,
                                                          voxel->subvoxel_values_[subvox_idx], 
                                                          LayerType::SUBVOXEL);
                    }
                }
            }
        }
    }
    
    // 处理新空闲的块，逻辑与上方类似
    for (const auto& block_idx : new_free_) {
        if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
            continue;
        }
        
        Block* block = blocks_[block_idx];
        Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
        Eigen::Vector3d block_center;
        blockIdxToWorld(block_grid_idx, block_center);
        
        // 根据层级添加不同分辨率的空闲体素
        if (block->layer_ == LayerType::BLOCK) {
            // 整个块作为单个空闲单元
            new_freed_voxels_.emplace_back(block_center, block_res_, 
                                          block->occupancy_value_, LayerType::BLOCK);
        }
        else if (block->layer_ == LayerType::VOXEL && block->is_voxel_allocated_) {
            // 添加块中的空闲体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || !voxel->is_free_) {
                    continue;
                }
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                Eigen::Vector3d voxel_center;
                voxelIdxToWorld(voxel_grid_idx, voxel_center);
                
                new_freed_voxels_.emplace_back(voxel_center, voxel_res_, 
                                              voxel->occupancy_value_, LayerType::VOXEL);
            }
        }
        else if (block->layer_ == LayerType::SUBVOXEL && block->is_voxel_allocated_) {
            // 添加块中的空闲子体素
            for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
                Voxel* voxel = block->voxels_[vox_idx];
                if (voxel == nullptr || !voxel->is_subvoxel_allocated_) {
                    continue;
                }
                
                Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
                
                if (voxel->is_free_) {
                    // 如果体素是空闲的，收集其中所有空闲的子体素
                    for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
                        if (voxel->subvoxel_values_[subvox_idx] < min_occupancy_log_) {
                            // 子体素是空闲的
                            Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
                            Eigen::Vector3d subvoxel_center;
                            subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);
                            
                            new_freed_voxels_.emplace_back(subvoxel_center, sub_voxel_res_,
                                                          voxel->subvoxel_values_[subvox_idx], 
                                                          LayerType::SUBVOXEL);
                        }
                    }
                }
            }
        }
    }
    
    // std::cout << "Collected " << new_occupied_voxels_.size() << " occupied voxels and " 
    //           << new_freed_voxels_.size() << " freed voxels at multiple resolutions." << std::endl;
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
        msg.size = voxel.size;
        msg.occupancy_value = voxel.occupancy_value;
        msg.layer = static_cast<uint8_t>(voxel.layer);
        
        msg_array.push_back(msg);
    }
}

// 单线程处理
// void SOGMMap::voxelPolarProjectionProcessWithRaycast(const cv::Mat &depth_image, 
//                                                   const Eigen::Matrix3d &R_C_2_W, 
//                                                   const Eigen::Vector3d &T_C_2_W) {
//     Eigen::Matrix3d R_W_2_C = R_C_2_W.transpose();
//     Eigen::Vector3d T_W_2_C = -R_W_2_C * T_C_2_W;

//     new_occ_.clear();
//     new_free_.clear();

//     // 处理自由空间块
//     for (size_t i = 0; i < free_blocks_.size(); ++i) {
//         int block_idx = free_blocks_[i];
//         if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
//             continue;
//         }
        
//         Block* block = blocks_[block_idx];

//         // 记录更新前的状态
//         bool was_occupied = !block->is_free_;

//         // 获取块的世界坐标
//         Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
//         Eigen::Vector3d block_center;
//         blockIdxToWorld(block_grid_idx, block_center);
        
//         // 计算块到相机的距离
//         Eigen::Vector3d block_camera = R_W_2_C * block_center + T_W_2_C;
//         double distance = block_camera.norm();
        
//         // 根据距离调整层级
//         switchLayer(block_idx, T_C_2_W);
        
//         // 重新获取块引用（因为switchLayer可能已修改了块）
//         block = blocks_[block_idx];
        
//         // 根据层级更新占据概率
//         if (block->layer_ == LayerType::BLOCK) {
//              updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
//         } 
//         else if (block->layer_ == LayerType::VOXEL) {
//             // 对每个体素更新概率
//             if (block->is_voxel_allocated_) {
//                 for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                     Voxel* voxel = block->voxels_[vox_idx];
//                     if (voxel != nullptr) {
//                         updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_miss_log_);
//                     }
//                 }

//                 // 向上传递更新后的占据信息
//                 propagateOccupancyUp(block);
//             }
//         }
//         else if (block->layer_ == LayerType::SUBVOXEL) {
//             // 对子体素进行更新
//             if (block->is_voxel_allocated_) {
//                 for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                     Voxel* voxel = block->voxels_[vox_idx];
//                     if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
//                         // 更新所有子体素
//                         for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
//                             // 直接使用子体素值更新（没有is_free标志）
//                             float &value = voxel->subvoxel_values_[subvox_idx];
//                             float new_value = value + prob_miss_log_;
//                             value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
//                         }
//                     }
//                 }
                
//                 // 向上传递更新后的占据信息
//                 propagateOccupancyUp(block);
//             }
//         }
//         // 检查状态是否从占据变为空闲
//         if (was_occupied && block->is_free_) {
//             new_free_.push_back(block_idx);
//         }
//     }
    
//     // 处理占据块部分
//     for (size_t i = 0; i < occ_blocks_.size(); ++i) {
//         int block_idx = occ_blocks_[i];
//         if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
//             continue;
//         }
        
//         Block* block = blocks_[block_idx];

//         // 记录更新前的状态
//         bool was_free = block->is_free_;
        
//         // 获取块的世界坐标
//         Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
//         Eigen::Vector3d block_center;
//         blockIdxToWorld(block_grid_idx, block_center);
        
//         // 根据距离调整层级
//         switchLayer(block_idx, T_C_2_W);
        
//         // 重新获取块引用（因为switchLayer可能已修改了块）
//         block = blocks_[block_idx];
        
//         // 根据层级进行处理
//         bool block_updated = false;
        
//         if (block->layer_ == LayerType::BLOCK) {
//             // Block层级：直接增加占据概率
//             updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
//             block_updated = true;
//         } 
//         else if (block->layer_ == LayerType::VOXEL) {
//             // Voxel层级：将块中所有voxel投影到深度图中
//             if (block->is_voxel_allocated_) {
//                 for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                     Voxel* voxel = block->voxels_[vox_idx];
//                     if (voxel == nullptr) continue;
                    
//                     // 获取体素的世界坐标
//                     Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
//                     Eigen::Vector3d voxel_center;
//                     voxelIdxToWorld(voxel_grid_idx, voxel_center);
                    
//                     // 投影体素到深度图并检查是否匹配
//                     if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                voxel_center, voxel_radius_)) {
//                         // 更新体素的占据概率
//                         updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
//                         block_updated = true;
//                     }
//                 }
                
//                 // 如果任何体素被更新，向上传递更新块的状态
//                 if (block_updated) {
//                     propagateOccupancyUp(block);
//                 }
//             }
//         } 
//         else if (block->layer_ == LayerType::SUBVOXEL) {
//             // Subvoxel层级：检查体素级别的投影，然后更新子体素
//             if (block->is_voxel_allocated_) {
//                 for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                     Voxel* voxel = block->voxels_[vox_idx];
//                     if (voxel == nullptr || !voxel->is_subvoxel_allocated_) {
//                         continue;
//                     }
                    
//                     // 获取体素的世界坐标
//                     Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
//                     Eigen::Vector3d voxel_center;
//                     voxelIdxToWorld(voxel_grid_idx, voxel_center);
                    
//                     // 投影体素到深度图并检查是否匹配
//                     if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                voxel_center, voxel_radius_)) {
//                         // 处理子体素级别的投影
//                         bool any_subvoxel_updated = false;
//                         // 遍历体素内的所有子体素
//                         for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
//                             if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
//                                 Eigen::Vector3i subvoxel_grid_idx = localLinearToVoxelIdx(subvox_idx, voxel_grid_idx);
//                                 Eigen::Vector3d subvoxel_center;
//                                 subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

//                                 // 投影子体素到深度图并检查是否匹配
//                                 if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                          subvoxel_center, voxel_radius_)) {
//                                     float &value = voxel->subvoxel_values_[subvox_idx];
//                                     float new_value = value + prob_hit_log_;
//                                     value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
//                                     any_subvoxel_updated = true;
//                                 }
//                             }
//                         }
//                         // 如果有任何子体素被更新，向上传递更新
//                         if (any_subvoxel_updated) {
//                             block_updated = true;
//                         }
//                     }
//                 }
                
//                 // 如果任何体素被更新，向上传递更新块的状态
//                 if (block_updated) {
//                     propagateOccupancyUp(block);
//                 }
//             }
//         }
//         // 检查状态是否从空闲变为占据
//         if (was_free && !block->is_free_) {
//             new_occ_.push_back(block_idx);
//         }
//     }

//     // 清空处理过的块列表
//     free_blocks_.clear();
//     occ_blocks_.clear();
// }

// 从大到小变换layer的时候，小单元直接继承大单元概率，导致可视化中一个块中的小单元全是占据
// void SOGMMap::voxelPolarProjectionProcessWithRaycast(const cv::Mat &depth_image, 
//                                                   const Eigen::Matrix3d &R_C_2_W, 
//                                                   const Eigen::Vector3d &T_C_2_W) {
//     Eigen::Matrix3d R_W_2_C = R_C_2_W.transpose();
//     Eigen::Vector3d T_W_2_C = -R_W_2_C * T_C_2_W;

//     new_occ_.clear();
//     new_free_.clear();

//     // 多线程处理的互斥锁，用于保护共享资源
//     std::mutex new_free_mutex;
//     std::mutex new_occ_mutex;

//     // 为多线程处理准备线程本地变量容器
//     std::vector<std::vector<int>> thread_new_free(num_projection_threads_);
//     std::vector<std::vector<int>> thread_new_occ(num_projection_threads_);

//     // 1. 处理自由空间块 - 多线程
//     int free_count = free_blocks_.size();
//     int blocks_per_thread = (free_count + num_projection_threads_ - 1) / num_projection_threads_;
    
//     std::vector<std::thread> free_threads;
//     free_threads.reserve(num_projection_threads_);
    
//     for (int t = 0; t < num_projection_threads_; ++t) {
//         size_t start_idx = t * blocks_per_thread;
//         size_t end_idx = std::min(start_idx + blocks_per_thread, static_cast<size_t>(free_count));
        
//         // 如果这个线程没有要处理的块，则跳过
//         if (start_idx >= end_idx) {
//             continue;
//         }
        
//         free_threads.emplace_back([this, &thread_new_free, &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
//             for (size_t i = start_idx; i < end_idx; ++i) {
//                 int block_idx = free_blocks_[i];
//                 if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
//                     continue;
//                 }
                
//                 Block* block = blocks_[block_idx];

//                 // 记录更新前的状态
//                 bool was_occupied = !block->is_free_;
//                 uint8_t free_voxels=0;
//                 uint8_t free_subvoxels=0;

//                 // 获取块的世界坐标
//                 Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
//                 Eigen::Vector3d block_center;
//                 blockIdxToWorld(block_grid_idx, block_center);
                
//                 // 计算块到相机的距离
//                 Eigen::Vector3d block_camera = R_W_2_C * block_center + T_W_2_C;
//                 double distance = block_camera.norm();
                
//                 // 根据距离调整层级 - 这里需要加锁
//                 {
//                     std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
//                     switchLayer(block_idx, T_C_2_W);
                    
//                     // 重新获取块引用（因为switchLayer可能已修改了块）
//                     block = blocks_[block_idx];
                    
//                     // 根据层级更新占据概率
//                     if (block->layer_ == LayerType::BLOCK) {
//                         updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_miss_log_);
//                     } 
//                     else if (block->layer_ == LayerType::VOXEL) {
//                         // 对每个体素更新概率
//                         if (block->is_voxel_allocated_) {
//                             for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                                 Voxel* voxel = block->voxels_[vox_idx];
//                                 if (voxel != nullptr) {
//                                     updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_miss_log_);
//                                     if (voxel->is_free_) {
//                                         free_voxels++;
//                                     }
//                                 }
//                             }

//                             // 向上传递更新后的占据信息
//                             propagateOccupancyUp(block);
//                         }
//                     }
//                     else if (block->layer_ == LayerType::SUBVOXEL) {
//                         // 对子体素进行更新
//                         if (block->is_voxel_allocated_) {
//                             for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                                 Voxel* voxel = block->voxels_[vox_idx];
//                                 if (voxel != nullptr && voxel->is_subvoxel_allocated_) {
//                                     // 更新所有子体素
//                                     for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
//                                         // 直接使用子体素值更新（没有is_free标志）
//                                         float &value = voxel->subvoxel_values_[subvox_idx];
//                                         float new_value = value + prob_miss_log_;
//                                         value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
//                                         if (value < min_occupancy_log_) {
//                                             free_subvoxels++;
//                                         }
//                                     }
//                                 }
//                             }
                            
//                             // 向上传递更新后的占据信息
//                             propagateOccupancyUp(block);
//                         }
//                     }
                    
//                     // 检查状态是否从占据变为空闲
//                     if ((was_occupied && block->is_free_) ||
//                         (free_voxels >= total_voxel_in_block_ /4) ||
//                         (free_subvoxels >= total_subvoxel_in_voxel_ /4)) {
//                         thread_new_free[t].push_back(block_idx);
//                     }
//                 }
//             }
//         });
//     }
    
//     // 等待所有自由空间线程完成
//     for (auto& thread : free_threads) {
//         if (thread.joinable()) {
//             thread.join();
//         }
//     }
    
//     // 2. 处理占据块部分 - 多线程
//     int occ_count = occ_blocks_.size();
//     blocks_per_thread = (occ_count + num_projection_threads_ - 1) / num_projection_threads_;
    
//     std::vector<std::thread> occ_threads;
//     occ_threads.reserve(num_projection_threads_);
    
//     for (int t = 0; t < num_projection_threads_; ++t) {
//         size_t start_idx = t * blocks_per_thread;
//         size_t end_idx = std::min(start_idx + blocks_per_thread, static_cast<size_t>(occ_count));
        
//         // 如果这个线程没有要处理的块，则跳过
//         if (start_idx >= end_idx) {
//             continue;
//         }
        
//         occ_threads.emplace_back([this, &thread_new_occ, &depth_image, &R_W_2_C, &T_W_2_C, &T_C_2_W, start_idx, end_idx, t]() {
//             for (size_t i = start_idx; i < end_idx; ++i) {
//                 int block_idx = occ_blocks_[i];
//                 if (block_idx < 0 || block_idx >= blocks_.size() || blocks_[block_idx] == nullptr) {
//                     continue;
//                 }
                
//                 // 加锁保护块访问
//                 {
//                     std::lock_guard<std::mutex> lock(block_mutex_[block_idx % NUM_BLOCK_MUTEXES]);
//                     Block* block = blocks_[block_idx];

//                     // 记录更新前的状态
//                     bool was_free = block->is_free_;
                    
//                     // 获取块的世界坐标
//                     Eigen::Vector3i block_grid_idx = linearToBlockIdx(block_idx);
//                     Eigen::Vector3d block_center;
//                     blockIdxToWorld(block_grid_idx, block_center);
                    
//                     // 根据距离调整层级
//                     switchLayer(block_idx, T_C_2_W);
                    
//                     // 重新获取块引用（因为switchLayer可能已修改了块）
//                     block = blocks_[block_idx];
                    
//                     // 根据层级进行处理
//                     bool block_updated = false;
                    
//                     if (block->layer_ == LayerType::BLOCK) {
//                         // Block层级：直接增加占据概率
//                         updateOccupancyValue(block->occupancy_value_, block->is_free_, prob_hit_log_);
//                         block_updated = true;
//                     } 
//                     else if (block->layer_ == LayerType::VOXEL) {
//                         // Voxel层级：将块中所有voxel投影到深度图中
//                         if (block->is_voxel_allocated_) {
//                             for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                                 Voxel* voxel = block->voxels_[vox_idx];
//                                 if (voxel == nullptr) continue;
                                
//                                 // 获取体素的世界坐标
//                                 Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
//                                 Eigen::Vector3d voxel_center;
//                                 voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                
//                                 // 投影体素到深度图并检查是否匹配
//                                 if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                             voxel_center, voxel_radius_)) {
//                                     // 更新体素的占据概率
//                                     updateOccupancyValue(voxel->occupancy_value_, voxel->is_free_, prob_hit_log_);
//                                     block_updated = true;
//                                 }
//                             }
                            
//                             // 如果任何体素被更新，向上传递更新块的状态
//                             if (block_updated) {
//                                 propagateOccupancyUp(block);
//                             }
//                         }
//                     } 
//                     else if (block->layer_ == LayerType::SUBVOXEL) {
//                         // Subvoxel层级：检查体素级别的投影，然后更新子体素
//                         if (block->is_voxel_allocated_) {
//                             for (int vox_idx = 0; vox_idx < block->voxels_.size(); ++vox_idx) {
//                                 Voxel* voxel = block->voxels_[vox_idx];
//                                 if (voxel == nullptr || !voxel->is_subvoxel_allocated_) {
//                                     continue;
//                                 }
                                
//                                 // 获取体素的世界坐标
//                                 Eigen::Vector3i voxel_grid_idx = localLinearToVoxelIdx(vox_idx, block_grid_idx);
//                                 Eigen::Vector3d voxel_center;
//                                 voxelIdxToWorld(voxel_grid_idx, voxel_center);
                                
//                                 // 投影体素到深度图并检查是否匹配
//                                 if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                            voxel_center, voxel_radius_)) {
//                                     // 处理子体素级别的投影
//                                     bool any_subvoxel_updated = false;
//                                     // 遍历体素内的所有子体素
//                                     for (int subvox_idx = 0; subvox_idx < voxel->subvoxel_values_.size(); ++subvox_idx) {
//                                         if (voxel->is_subvoxel_allocated_) {
//                                             Eigen::Vector3i subvoxel_grid_idx = localLinearToSubVoxelIdx(subvox_idx, voxel_grid_idx);
//                                             Eigen::Vector3d subvoxel_center;
//                                             subVoxelIdxToWorld(subvoxel_grid_idx, subvoxel_center);

//                                             // 投影子体素到深度图并检查是否匹配
//                                             if (projectVoxelToDepthImage(depth_image, R_W_2_C, T_W_2_C, 
//                                                                      subvoxel_center, sub_voxel_radius_)) {
//                                                 float &value = voxel->subvoxel_values_[subvox_idx];
//                                                 float new_value = value + prob_hit_log_;
//                                                 value = std::min(std::max(new_value, clamp_min_log_), clamp_max_log_);
//                                                 any_subvoxel_updated = true;
//                                             }
//                                         }
//                                     }
//                                     // 如果有任何子体素被更新，向上传递更新
//                                     if (any_subvoxel_updated) {
//                                         block_updated = true;
//                                     }
//                                 }
//                             }
                            
//                             // 如果任何体素被更新，向上传递更新块的状态
//                             if (block_updated) {
//                                 propagateOccupancyUp(block);
//                             }
//                         }
//                     }
                    
//                     // 检查状态是否从空闲变为占据
//                     if (was_free && !block->is_free_) {
//                         thread_new_occ[t].push_back(block_idx);
//                     }
//                 }
//             }
//         });
//     }
    
//     // 等待所有占据块线程完成
//     for (auto& thread : occ_threads) {
//         if (thread.joinable()) {
//             thread.join();
//         }
//     }
    
//     // 合并所有线程结果到主结果容器
//     for (int t = 0; t < num_projection_threads_; ++t) {
//         new_free_.insert(new_free_.end(), thread_new_free[t].begin(), thread_new_free[t].end());
//         new_occ_.insert(new_occ_.end(), thread_new_occ[t].begin(), thread_new_occ[t].end());
//     }

//     // std::cout << "Processed " << new_occ_.size() << " occupied blocks and " 
//     //           << new_free_.size() << " free blocks." << std::endl;
//     // 收集多层级的体素
//     collectMultiLayerVoxels();

//     // 清空处理过的块列表
//     free_blocks_.clear();
//     occ_blocks_.clear();
// }