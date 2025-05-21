# 单个多层级体素信息
geometry_msgs/Point position  # 体素中心位置
float32 size                  # 体素尺寸(立方体边长)
float32 occupancy_value       # 占据值
uint8 layer                   # 层级类型: 0=BLOCK, 1=VOXEL, 2=SUBVOXEL