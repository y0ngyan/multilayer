#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <multilayer/VoxelGridMsg.h>
#include <multilayer/VoxelGridMsgArray.h>

// 为每个分辨率层级创建一个全局Publisher
ros::Publisher block_pub;
ros::Publisher voxel_pub;
ros::Publisher subvoxel_pub;
std::string frame_id = "map";

// 订阅 /map/local_state 话题的回调函数
void mapStateCallback(const multilayer::VoxelGridMsgArrayConstPtr& msg)
{
    // 为每个层级创建独立的PCL点云
    pcl::PointCloud<pcl::PointXYZ> block_cloud;
    pcl::PointCloud<pcl::PointXYZ> voxel_cloud;
    pcl::PointCloud<pcl::PointXYZ> subvoxel_cloud;

    // 根据收到的消息填充对应的点云
    for (const auto& voxel_msg : msg->voxels)
    {
        pcl::PointXYZ pt;
        pt.x = voxel_msg.position.x;
        pt.y = voxel_msg.position.y;
        pt.z = voxel_msg.position.z;

        switch (voxel_msg.layer)
        {
            case 0: // BLOCK - 最粗糙
                block_cloud.points.push_back(pt);
                break;
            case 1: // VOXEL - 中等
                voxel_cloud.points.push_back(pt);
                break;
            case 2: // SUBVOXEL - 最精细
                subvoxel_cloud.points.push_back(pt);
                break;
        }
    }

    // 准备三个独立的ROS消息进行发布
    sensor_msgs::PointCloud2 block_msg, voxel_msg, subvoxel_msg;

    // 1. 处理并发布Block层级的点云
    pcl::toROSMsg(block_cloud, block_msg);
    block_msg.header = msg->header;
    block_pub.publish(block_msg);

    // 2. 处理并发布Voxel层级的点云
    pcl::toROSMsg(voxel_cloud, voxel_msg);
    voxel_msg.header = msg->header;
    voxel_pub.publish(voxel_msg);

    // 3. 处理并发布Sub-voxel层级的点云
    pcl::toROSMsg(subvoxel_cloud, subvoxel_msg);
    subvoxel_msg.header = msg->header;
    subvoxel_pub.publish(subvoxel_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "local_map_visualizer_node");
    ros::NodeHandle nh("~");

    nh.param<std::string>("frame_id", frame_id, "map");

    // 初始化三个独立的Publisher
    block_pub = nh.advertise<sensor_msgs::PointCloud2>("/map/local_vis/blocks", 10);
    voxel_pub = nh.advertise<sensor_msgs::PointCloud2>("/map/local_vis/voxels", 10);
    subvoxel_pub = nh.advertise<sensor_msgs::PointCloud2>("/map/local_vis/subvoxels", 10);

    // 订阅局部地图状态话题
    ros::Subscriber map_state_sub = nh.subscribe("/map/local_state", 10, mapStateCallback);

    ROS_INFO("Local map visualizer started. Subscribing to /map/local_state.");
    ROS_INFO("Publishing separate topics for each resolution layer: /map/local_vis/{blocks, voxels, subvoxels}");
    
    ros::spin();
    return 0;
}