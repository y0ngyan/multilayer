//
//  RayCast.cpp
//  uav
//
//  Created by 李勇 on 2023/3/8.
//

#include "RayCast.hpp"
#include <limits> // Required for std::numeric_limits

// 用于计算一个整数的符号函数
int RayCaster::signum(int x)
{
  return x == 0 ? 0 : x < 0 ? -1
                            : 1;
}

// 计算 value 对 modulus 取模的非负结果
// 若value为负数，则fmod(value, modulus)结果为负数，此时加上modulus使结果为非负数
double RayCaster::mod(double value, double modulus)
{
  return fmod(fmod(value, modulus) + modulus, modulus);
}

// 计算s+ds*t为整数的最小正整数t
double RayCaster::intbound(double s, double ds)
{
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0)
  {
    return intbound(-s, -ds);
  }
  else
  {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

bool RayCaster::setInput(const Eigen::Vector3d &start,
                         const Eigen::Vector3d &end /* , const Eigen::Vector3d& min,
                         const Eigen::Vector3d& max */
)
{
  start_ = start;
  end_ = end;

  // 向下取整
  x_ = (int)std::floor(start_.x());
  y_ = (int)std::floor(start_.y());
  z_ = (int)std::floor(start_.z());
  endX_ = (int)std::floor(end_.x());
  endY_ = (int)std::floor(end_.y());
  endZ_ = (int)std::floor(end_.z());
  // direction_ = (end_ - start_);
  // maxDist_ = direction_.squaredNorm();

  // 方向向量
  // Break out direction vector.
  dx_ = endX_ - x_;
  dy_ = endY_ - y_;
  dz_ = endZ_ - z_;

  // 单位方向向量
  // Direction to increment x,y,z when stepping.
  stepX_ = signum(dx_);
  stepY_ = signum(dy_);
  stepZ_ = signum(dz_);

  // 计算从起点到下一个格子边界所需的参数 t 值
  // See description above. The initial values depend on the fractional
  // part of the origin.
  tMaxX_ = intbound(start_.x(), dx_);
  tMaxY_ = intbound(start_.y(), dy_);
  tMaxZ_ = intbound(start_.z(), dz_);

  // 计算在每个轴向上，跨越一个格子所需增加的 t 值
  // The change in t when taking a step (always positive).
  tDeltaX_ = (dx_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepX_ / dx_);
  tDeltaY_ = (dy_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepY_ / dy_);
  tDeltaZ_ = (dz_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepZ_ / dz_);

  // dist_ = 0;

  // step_num_ = 0;

  // Avoids an infinite loop.
  if (stepX_ == 0 && stepY_ == 0 && stepZ_ == 0)
    return false;
  else
    return true;
}

// 用于迭代光线投射过程，获取下一个经过的格子坐标
bool RayCaster::step(Eigen::Vector3d &ray_pt)
{
  // if (x_ >= min_.x() && x_ < max_.x() && y_ >= min_.y() && y_ < max_.y() &&
  // z_ >= min_.z() && z_ <
  // max_.z())
  ray_pt = Eigen::Vector3d(x_, y_, z_);

  // step_num_++;

  // dist_ = (Eigen::Vector3d(x_, y_, z_) - start_).squaredNorm();

  if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
  {
    return false;
  }

  // if (dist_ > maxDist_)
  // {
  //   return false;
  // }

  // tMaxX stores the t-value at which we cross a cube boundary along the
  // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
  // chooses the closest cube boundary. Only the first case of the four
  // has been commented in detail.
  // 虽然x_已经向下取整，但是假设起点并不严格在格子边界
  // tMaxX_记录的是从起点到第一个格子边界所需的时间
  // 选择最小的tMax，即选择最靠近格子边界的轴，先进行移动
  // 更新tMax，不断迭代，记录从起点到终点的t值
  if (tMaxX_ < tMaxY_)
  {
    if (tMaxX_ < tMaxZ_)
    {
      // Update which cube we are now in.
      x_ += stepX_;
      // Adjust tMaxX to the next X-oriented boundary crossing.
      tMaxX_ += tDeltaX_;
    }
    else
    {
      z_ += stepZ_;
      tMaxZ_ += tDeltaZ_;
    }
  }
  else
  {
    if (tMaxY_ < tMaxZ_)
    {
      y_ += stepY_;
      tMaxY_ += tDeltaY_;
    }
    else
    {
      z_ += stepZ_;
      tMaxZ_ += tDeltaZ_;
    }
  }

  return true;
}

// 在光线终点前留出一定的最小距离 min，避免光线直接达到终点
bool RayCaster::setInput(const Eigen::Vector3d &start,
                         const Eigen::Vector3d &end, const double &min_dist_to_end)
{

  Eigen::Vector3d delta = end - start;
  // 避免 delta 长度为0的情况
  if (delta.norm() < 1e-9) { // 如果起点和终点非常接近
    end_ = start_; // 将终点设为起点，后续逻辑会处理
  } else {
    end_ = end - min_dist_to_end / delta.norm() * delta;
  }

  start_ = start;
  // end_ = end;

  x_ = (int)std::floor(start_.x());
  y_ = (int)std::floor(start_.y());
  z_ = (int)std::floor(start_.z());
  endX_ = (int)std::floor(end_.x());
  endY_ = (int)std::floor(end_.y());
  endZ_ = (int)std::floor(end_.z());
  // direction_ = (end_ - start_);
  // maxDist_ = direction_.squaredNorm();

  // Break out direction vector.
  dx_ = endX_ - x_;
  dy_ = endY_ - y_;
  dz_ = endZ_ - z_;

  // Direction to increment x,y,z when stepping.
  stepX_ = signum(dx_);
  stepY_ = signum(dy_);
  stepZ_ = signum(dz_);

  // See description above. The initial values depend on the fractional
  // part of the origin.
  tMaxX_ = intbound(start_.x(), dx_);
  tMaxY_ = intbound(start_.y(), dy_);
  tMaxZ_ = intbound(start_.z(), dz_);

  // The change in t when taking a step (always positive).
  tDeltaX_ = (dx_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepX_ / dx_);
  tDeltaY_ = (dy_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepY_ / dy_);
  tDeltaZ_ = (dz_ == 0) ? std::numeric_limits<double>::infinity() : ((double)stepZ_ / dz_);

  // dist_ = 0;

  // step_num_ = 0;

  // Avoids an infinite loop.
  if (stepX_ == 0 && stepY_ == 0 && stepZ_ == 0)
    return false;
  else
    return true;
}

// 步进函数重载，输入参数为Eigen::Vector3i类型
bool RayCaster::step(Eigen::Vector3i &ray_pt)
{
  // if (x_ >= min_.x() && x_ < max_.x() && y_ >= min_.y() && y_ < max_.y() &&
  // z_ >= min_.z() && z_ <
  // max_.z())
  ray_pt = Eigen::Vector3i(x_, y_, z_);

  // step_num_++;

  // dist_ = (Eigen::Vector3d(x_, y_, z_) - start_).squaredNorm();

  if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
  {
    return false;
  }

  // if (dist_ > maxDist_)
  // {
  //   return false;
  // }

  // tMaxX stores the t-value at which we cross a cube boundary along the
  // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
  // chooses the closest cube boundary. Only the first case of the four
  // has been commented in detail.
  if (tMaxX_ < tMaxY_)
  {
    if (tMaxX_ < tMaxZ_)
    {
      // Update which cube we are now in.
      x_ += stepX_;
      // Adjust tMaxX to the next X-oriented boundary crossing.
      tMaxX_ += tDeltaX_;
    }
    else
    {
      z_ += stepZ_;
      tMaxZ_ += tDeltaZ_;
    }
  }
  else
  {
    if (tMaxY_ < tMaxZ_)
    {
      y_ += stepY_;
      tMaxY_ += tDeltaY_;
    }
    else
    {
      z_ += stepZ_;
      tMaxZ_ += tDeltaZ_;
    }
  }

  return true;
}

// 跳跃函数实现 - 浮点数版本
bool RayCaster::jump(Eigen::Vector3d &ray_pt, int multiple_steps)
{
    if (multiple_steps <= 0) {
        return false;
    }
    
    // 先记录当前位置
    ray_pt = Eigen::Vector3d(x_, y_, z_);
    
    // 如果已到达终点，直接返回
    if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
    {
        return false;
    }
    
    // 执行多次步进
    for (int i = 0; i < multiple_steps; ++i)
    {
        // 选择tMax最小的轴进行步进
        if (tMaxX_ < tMaxY_)
        {
            if (tMaxX_ < tMaxZ_)
            {
                x_ += stepX_;
                tMaxX_ += tDeltaX_;
            }
            else
            {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        }
        else
        {
            if (tMaxY_ < tMaxZ_)
            {
                y_ += stepY_;
                tMaxY_ += tDeltaY_;
            }
            else
            {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        }
        
        // 检查是否到达终点
        if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
        {
            ray_pt = Eigen::Vector3d(x_, y_, z_);
            return false;
        }
    }
    
    // 更新输出的光线点位置
    ray_pt = Eigen::Vector3d(x_, y_, z_);
    return true;
}

// 跳跃函数实现 - 整数版本
bool RayCaster::jump(Eigen::Vector3i &ray_pt, int multiple_steps)
{
    if (multiple_steps <= 0) {
        return false;
    }
    
    // 先记录当前位置
    ray_pt = Eigen::Vector3i(x_, y_, z_);
    
    // 如果已到达终点，直接返回
    if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
    {
        return false;
    }
    
    // 执行多次步进
    for (int i = 0; i < multiple_steps; ++i)
    {
        // 选择tMax最小的轴进行步进
        if (tMaxX_ < tMaxY_)
        {
            if (tMaxX_ < tMaxZ_)
            {
                x_ += stepX_;
                tMaxX_ += tDeltaX_;
            }
            else
            {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        }
        else
        {
            if (tMaxY_ < tMaxZ_)
            {
                y_ += stepY_;
                tMaxY_ += tDeltaY_;
            }
            else
            {
                z_ += stepZ_;
                tMaxZ_ += tDeltaZ_;
            }
        }
        
        // 检查是否到达终点
        if (x_ == endX_ && y_ == endY_ && z_ == endZ_)
        {
            ray_pt = Eigen::Vector3i(x_, y_, z_);
            return false;
        }
    }
    
    // 更新输出的光线点位置
    ray_pt = Eigen::Vector3i(x_, y_, z_);
    return true;
}

// 计算从当前位置到目标位置的最大安全步数
int RayCaster::calculateSafeSteps(const Eigen::Vector3d &target_pos)
{
    // 计算目标格子坐标
    int target_x = static_cast<int>(std::floor(target_pos.x()));
    int target_y = static_cast<int>(std::floor(target_pos.y()));
    int target_z = static_cast<int>(std::floor(target_pos.z()));
    
    // 计算当前点到目标点的曼哈顿距离
    int manhattan_dist = std::abs(target_x - x_) + std::abs(target_y - y_) + std::abs(target_z - z_);
    
    // 安全步数取曼哈顿距离的一半，确保不会跳过终点
    return std::max(1, manhattan_dist / 2);
}

// 重置光线投射器的当前位置
void RayCaster::resetPosition(const Eigen::Vector3i &new_pos)
{
    x_ = new_pos.x();
    y_ = new_pos.y();
    z_ = new_pos.z();
    
    // 重新计算tMax值，考虑步进方向
    if (stepX_ != 0) {
        tMaxX_ = intbound(static_cast<double>(x_), static_cast<double>(stepX_));
    }
    if (stepY_ != 0) {
        tMaxY_ = intbound(static_cast<double>(y_), static_cast<double>(stepY_));
    }
    if (stepZ_ != 0) {
        tMaxZ_ = intbound(static_cast<double>(z_), static_cast<double>(stepZ_));
    }
}
