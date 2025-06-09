# multilayer

主要是参数问题，特别是将体素和子体素投影到深度图的半径大小，过大的话会导致不同分辨率之间的连接出现大量空洞
还有yaml中的比例阈值和深度差阈值，也有一点影响

fuck
还是深度差阈值影响最大，明天修改一下，如果最大值和最小值的差值过大，则取中心点数据

运行点云数据
roslaunch multilayer pointcloud_multilayer_map.launch
rosbag play HKUdense2.bag --topic /aft_mapped_to_init /depth_image /livox/pointcloud2

运行深度图数据
roslaunch multilayer multilayer_map.launch 
rosbag play uHumans2_apartment_s1_00h.bag --topic /tesse/odom /tesse/depth_cam/mono/image_raw

室内最好的一组参数
%YAML:1.2
---
OccMap:
  resolution: 0.05 # (unit: m)
  map_x: 40 # (unit: m)
  map_y: 40 # (unit: m)
  map_z: 30 # (unit: m)
  voxel_depth: 1
  block_depth: 3
  near_distance_threshold: 5
  far_distance_threshold: 10

  sub_radius_ratio: 0.3
  voxel_radius_ratio: 0.3
  block_radius_ratio: 0.5

Probability:
  p_hit: 0.7
  p_miss: 0.4
  p_min: 0.12
  p_max: 0.97
  p_occ: 0.75

DepthCamera:
  # uhuman
  height: 480
  width: 720
  fx: 415.69219381653056
  fy: 415.69219381653056
  cx: 360.0
  cy: 240.0

  k_depth_scaling_factor: 1
  depth_maxdist: 20.0
  depth_mindist: 0.1
  depth_filter_margin: 20
  skip_pixel: 4
  min_valid_ratio: 0.3
  depth_threshold_subvoxel: 0.05           # 体素z距离与像素平均值距离比较阈值
  depth_threshold_voxel: 0.1            # 体素z距离与体素平均值距离比较阈值
  depth_threshold_block: 0.2            # 体素z距离与体素平均值距离比较阈值

  R_C_2_B: !!opencv-matrix #normal realsense
    rows: 3
    cols: 3
    dt: d
    data: [0, 0, 1, -1, 0, 0, 0, -1, 0]

  T_C_2_B: !!opencv-matrix
    rows: 1
    cols: 3
    dt: d
    data: [0.0, 0.0, 0.0]


[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/y0ngyan/multilayer)
