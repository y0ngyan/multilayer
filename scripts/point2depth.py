#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

class Point2DepthConverter:
    def __init__(self):
        rospy.init_node('point2depth_converter', anonymous=True)
        
        # 从配置文件中读取的参数 (avia.yaml)
        self.fov_theta_range = [-35.2, 35.2]  # 水平视场角范围 (度)
        self.fov_phi_range = [-38.7, 38.7]    # 垂直视场角范围 (度)
        self.fov_depth = 60.0                  # 最大探测距离 (米)
        self.sensor_res_hor = 0.2            # 水平分辨率 (度)
        self.sensor_res_vert = 0.8             # 垂直分辨率 (度)
        
        # 计算深度图尺寸
        self.width = int((self.fov_theta_range[1] - self.fov_theta_range[0]) / self.sensor_res_hor) + 1
        self.height = int((self.fov_phi_range[1] - self.fov_phi_range[0]) / self.sensor_res_vert) + 1
        
        print(f"深度图尺寸: {self.width} x {self.height}")
        print(f"水平视场角: {self.fov_theta_range[0]}° 到 {self.fov_theta_range[1]}°")
        print(f"垂直视场角: {self.fov_phi_range[0]}° 到 {self.fov_phi_range[1]}°")
        print(f"最大深度: {self.fov_depth}m")
        
        # ROS相关
        self.bridge = CvBridge()
        # self.lidar_topic = rospy.get_param('~lidar_topic', '/cloud_registered')
        self.lidar_topic = rospy.get_param('~lidar_topic', '/livox/pointcloud2')
        
        # 订阅点云数据
        self.point_sub = rospy.Subscriber(self.lidar_topic, PointCloud2, self.point_cloud_callback)
        
        # 发布深度图
        self.depth_pub = rospy.Publisher('/depth_image', Image, queue_size=1)
        self.depth_vis_pub = rospy.Publisher('/depth_image_viz', Image, queue_size=1)
        
        # 计算相机内参
        self.K, self.fx, self.fy, self.cx, self.cy = self.calculate_camera_intrinsics()
        
        print(f"深度图内参:")
        print(f"fx: {self.fx:.2f}, fy: {self.fy:.2f}")
        print(f"cx: {self.cx:.2f}, cy: {self.cy:.2f}")
        print(f"内参矩阵 K:\n{self.K}")

        rospy.loginfo("Point2Depth转换器已启动")
    
    def calculate_camera_intrinsics(self):
        """计算深度图对应的相机内参"""
        # 将视场角转换为弧度
        fov_h_rad = np.radians(self.fov_theta_range[1] - self.fov_theta_range[0])
        fov_v_rad = np.radians(self.fov_phi_range[1] - self.fov_phi_range[0])
        
        # 使用针孔相机模型计算焦距
        # fx = width / (2 * tan(fov_h/2))
        # fy = height / (2 * tan(fov_v/2))
        fx = self.width / (2 * np.tan(fov_h_rad / 2))
        fy = self.height / (2 * np.tan(fov_v_rad / 2))
        
        # 主点设置为图像中心
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        
        # 构建内参矩阵
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        
        return K, fx, fy, cx, cy

    def point_cloud_callback(self, msg):
        try:
            # 转换点云数据
            points = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])
            
            if len(points) == 0:
                rospy.logwarn("接收到空点云数据")
                return
            
            points = np.array(points)
            
            # 转换为深度图
            depth_image = self.points_to_depth_image(points)
            
            # 发布深度图
            self.publish_depth_image(depth_image, msg.header)
            
        except Exception as e:
            rospy.logerr(f"处理点云数据时出错: {e}")
    
    def points_to_depth_image(self, points):
        """使用针孔相机模型将3D点云投影到深度图"""
        # 初始化深度图
        depth_image = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 激光雷达坐标系到相机坐标系的转换
        # 激光雷达: X前方, Y左方, Z上方
        # 相机: Z前方, X右方, Y下方
        # 转换关系: 
        # 相机X = -激光雷达Y (右方 = -左方)
        # 相机Y = -激光雷达Z (下方 = -上方) 
        # 相机Z = 激光雷达X (前方 = 前方)
        
        lidar_x, lidar_y, lidar_z = points[:, 0], points[:, 1], points[:, 2]
        
        # 坐标系转换
        camera_x = -lidar_y  # 相机右方 = 激光雷达左方的反方向
        camera_y = -lidar_z  # 相机下方 = 激光雷达上方的反方向
        camera_z = lidar_x   # 相机前方 = 激光雷达前方
        
        # 过滤距离超出范围的点
        distances = np.sqrt(lidar_x**2 + lidar_y**2 + lidar_z**2)
        valid_distance_mask = (distances > 0) & (distances <= self.fov_depth)
        
        camera_x_valid = camera_x[valid_distance_mask]
        camera_y_valid = camera_y[valid_distance_mask]
        camera_z_valid = camera_z[valid_distance_mask]
        valid_distances = distances[valid_distance_mask]
        
        if len(camera_z_valid) == 0:
            return depth_image
        
        # 过滤在相机前方的点（相机坐标系要求Z > 0）
        front_mask = camera_z_valid > 0
        front_x = camera_x_valid[front_mask]
        front_y = camera_y_valid[front_mask]
        front_z = camera_z_valid[front_mask]
        front_distances = valid_distances[front_mask]
        
        if len(front_z) == 0:
            return depth_image
        
        # 使用针孔相机模型投影3D点到2D像素坐标
        # 投影公式: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
        u = self.fx * (front_x / front_z) + self.cx
        v = self.fy * (front_y / front_z) + self.cy
        
        # 转换为整数像素坐标
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # 过滤在图像范围内的点
        image_mask = (u_int >= 0) & (u_int < self.width) & (v_int >= 0) & (v_int < self.height)
        
        valid_u = u_int[image_mask]
        valid_v = v_int[image_mask]
        valid_depths = front_z[image_mask]  # 使用相机坐标系的Z坐标作为深度值
        
        if len(valid_depths) == 0:
            print("没有有效的点投影到深度图中")
            return depth_image
        
        # 将深度值赋给对应像素
        # 如果同一像素有多个点，取最近的深度值
        for i in range(len(valid_depths)):
            current_depth = depth_image[valid_v[i], valid_u[i]]
            if current_depth == 0 or valid_depths[i] < current_depth:
                depth_image[valid_v[i], valid_u[i]] = valid_depths[i]
        
        return depth_image
    
    def publish_depth_image(self, depth_image, header):
        """发布深度图"""
        try:
            # 发布原始深度图 (32位浮点数，单位：米)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
            depth_msg.header = header
            # depth_msg.header.frame_id = "camera_init"
            self.depth_pub.publish(depth_msg)
            
            # 创建可视化深度图 (8位灰度图)
            depth_viz = self.create_depth_visualization(depth_image)
            depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="mono8")
            depth_viz_msg.header = header
            # depth_viz_msg.header.frame_id = "camera_init"
            self.depth_vis_pub.publish(depth_viz_msg)
            
            rospy.loginfo_throttle(1.0, f"已发布深度图，尺寸: {depth_image.shape}")
            
        except Exception as e:
            rospy.logerr(f"发布深度图时出错: {e}")
    
    def create_depth_visualization(self, depth_image):
        """创建深度图可视化"""
        # 归一化到0-255范围
        depth_viz = depth_image.copy()
        
        # 过滤零值
        valid_mask = depth_viz > 0
        if np.any(valid_mask):
            # 将深度值映射到0-255
            max_depth = min(np.max(depth_viz[valid_mask]), self.fov_depth)
            depth_viz[valid_mask] = (depth_viz[valid_mask] / max_depth * 255).astype(np.uint8)
            depth_viz[~valid_mask] = 0
        else:
            depth_viz = np.zeros_like(depth_viz, dtype=np.uint8)
        
        return depth_viz.astype(np.uint8)
    
    def run(self):
        """运行转换器"""
        rospy.loginfo("等待点云数据...")
        rospy.spin()

if __name__ == '__main__':
    try:
        converter = Point2DepthConverter()
        converter.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Point2Depth转换器已停止")
    except Exception as e:
        rospy.logerr(f"转换器运行出错: {e}")