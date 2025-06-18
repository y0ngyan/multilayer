# 单个多层级体素信息
geometry_msgs/Point position  # 体素中心位置
uint8 layer                   # 层级类型: 0=BLOCK, 1=VOXEL, 2=SUBVOXEL

# 是否被判定为高置信度地面块
bool is_ground
# 地面置信度的原始概率值 (0.0 ~ 1.0) - 为未来调试保留
float32 ground_confidence