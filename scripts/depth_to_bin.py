#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# --- 用户配置参数 ---

# TODO: 这是最重要的部分！请务必替换为您自己相机的真实内参。
# 如果没有准确的内参，生成的点云将是错误的。
CAMERA_INTRINSICS = {
    'fx': 415.69219381653056,  # X轴焦距
    'fy': 415.69219381653056,  # Y轴焦距
    'cx': 360.0,             # X轴主点
    'cy': 240.0              # Y轴主点
}

# TODO: 修改为您希望保存.bin文件的路径
SAVE_PATH = "/home/yy/ws_mrsogm/src/multilayer/pointcloud"

# --- 全局变量 ---
cv_bridge = CvBridge()
frame_count = 0

def depth_to_pointcloud(depth_image, intrinsics):
    """
    将深度图转换为点云 (N, 4)
    :param depth_image: 从ROS话题解码后的NumPy深度图
    :param intrinsics: 相机内参字典
    :return: NumPy数组形式的点云, 格式为 [X, Y, Z, Intensity]
    """
    height, width = depth_image.shape
    
    # 创建像素坐标网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 过滤掉无效的深度值 (例如0或NaN)
    valid_mask = (depth_image > 0) & np.isfinite(depth_image)
    
    # 将深度值展平
    z = depth_image[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]

    # 根据相机内参反向投影到3D空间
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    x = (u - intrinsics['cx']) * z / intrinsics['fx']
    y = (v - intrinsics['cy']) * z / intrinsics['fy']
    
    # 按照SPVNAS格式，为每个点添加强度值（这里默认为1.0）
    intensity = np.ones_like(z)

    # 将 X, Y, Z, Intensity 堆叠成 (N, 4) 的格式
    pointcloud = np.vstack((x, y, z, intensity)).T
    
    return pointcloud.astype(np.float32)

def image_callback(msg):
    """
    处理深度图ROS消息的回调函数
    """
    global frame_count
    rospy.loginfo("接收到一帧深度图像，序列号: %d" % msg.header.seq)

    try:
        # 将ROS图像消息转换为OpenCV格式 (NumPy数组)
        # 假设深度图是32位浮点数，单位是米。如果不是，需要修改encoding。
        # 例如，如果是16位无符号整数，单位是毫米，使用 "16UC1" 并对结果/1000.0
        depth_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # 将深度图转换为点云
    pointcloud = depth_to_pointcloud(depth_image, CAMERA_INTRINSICS)
    
    if pointcloud.size == 0:
        rospy.logwarn("生成的点云为空，可能深度图全为无效值。")
        return

    # 构造保存文件名并保存
    filename = os.path.join(SAVE_PATH, "%06d.bin" % frame_count)
    try:
        # 以二进制格式写入文件
        pointcloud.tofile(filename)
        rospy.loginfo("点云已保存至: %s, 包含 %d 个点" % (filename, pointcloud.shape[0]))
        frame_count += 1
    except Exception as e:
        rospy.logerr("保存文件失败: %s" % e)


def main():
    """
    主函数，初始化ROS节点和订阅者
    """
    # 检查保存路径是否存在
    if not os.path.exists(SAVE_PATH):
        rospy.loginfo("保存路径 %s 不存在，正在创建..." % SAVE_PATH)
        os.makedirs(SAVE_PATH)

    rospy.init_node('depth_to_bin_converter', anonymous=True)
    
    topic_name = "/tesse/depth_cam/mono/image_raw"
    rospy.Subscriber(topic_name, Image, image_callback, queue_size=10)
    
    rospy.loginfo("节点已启动，正在监听话题: %s" % topic_name)
    rospy.loginfo("生成的.bin文件将保存在: %s" % SAVE_PATH)
    
    # 保持节点运行直到被关闭
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass