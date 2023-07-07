#!/usr/bin/env python

import argparse

import torch
import open3d
import numpy as np
from ctypes import *  # convert float to uint32
import torch_geometric.data as tgd
import itertools
import typing


# ROS imports
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

# flowbot
from flowbot3d.datasets.flow_dataset_pyg import Flowbot3DPyGDataset, Flowbot3DTGData
# import flowbot3d.tg_dataset as ftd
import flowbot3d.models.artflownet as fmf


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8


def convert_rgbUint32_to_tuple(rgb_uint32): return (
    (rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8, (rgb_uint32 & 0x000000ff)
)


def convert_rgbFloat_to_tuple(rgb_float): return convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion


def convertCloudFromRosToOpen3d(ros_cloud):

    # Get cloud data from ros_cloud
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names))

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data) == 0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD = 3  # x, y, z, rgb

        # Get xyz
        xyz = [(x, y, z) for x, y, z, rgb in cloud_data]  # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD]) == float:  # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data]

        # rgb = 0.0*xyz

        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x, y, z) for x, y, z in cloud_data]  # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]
        colors = colors.astype(np.uint8)
        points_list = points.tolist()
        colors_list = colors.reshape(-1,1).tolist()
        cloud_data = list(map(list.__add__, points_list, colors_list))
        
    return pc2.create_cloud(header, fields, cloud_data)

def convertTensorToPointsArray(tensor):
    points = []
    for i in range(tensor.shape[0]):
        p = Point()
        p.x = tensor[i][0]
        p.y = tensor[i][1]
        p.z = tensor[i][2]
        points.append(p)

    return points


class RosFlowbot3d:

    def __init__(self, args):

        self.model = fmf.ArtFlowNet.load_from_checkpoint(args.model_path).cuda()
        self.model.eval()

        self.depth_image_queue = []
        self.queue_size = 10
        self.bridge = CvBridge()

        self.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=613.7716064453125, fy=614.099609375, cx=314.3857421875, cy=242.46636962890625)

        self.camera_k_matrix = open3d.core.Tensor(self.camera_intrinsics.intrinsic_matrix)
        # Publishers
        self.pointcloud_pub = rospy.Publisher('masked_pointcloud', PointCloud2, queue_size=10)
        self.flow_pub = rospy.Publisher('affordance', Marker, queue_size=10)
        # Subscribers
        rospy.Subscriber("/live_camera/aligned_depth_to_color/image_raw",
                         Image, self.depthImageCallback, queue_size=10)
        rospy.Subscriber("/sam_node/mask", Image, self.maskedImageCallback, queue_size=10)

    def maskedImageCallback(self, image_mask_msg):
        print("got masked image")
        # convert image mask to open3d
        image_mask_opencv = self.bridge.imgmsg_to_cv2(image_mask_msg, desired_encoding='32FC1')
        # image_mask_opencv = np.array(image_mask_opencv,order="C") #not sure why this is needed
        image_mask_open3d = open3d.t.geometry.Image(image_mask_opencv)

        # convert depth image to open3d
        latest_depth_image_msg = self.depth_image_queue.pop()
        depth_img_opencv = self.bridge.imgmsg_to_cv2(latest_depth_image_msg, desired_encoding='32FC1')
        # depth_img_opencv = np.array(depth_img_opencv,order="C") #not sure why this is needed
        depth_img_open3d = open3d.t.geometry.Image(depth_img_opencv)

        # create rgbd
        rgbd_image = open3d.t.geometry.RGBDImage(image_mask_open3d, depth_img_open3d)
        point_cloud_open3d = open3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.camera_k_matrix)
        point_cloud_msg = convertCloudFromOpen3dToRos(point_cloud_open3d.to_legacy(), frame_id=latest_depth_image_msg.header.frame_id)
        self.pointcloud_pub.publish(point_cloud_msg)
        
        # pointcloud_open3d = convertCloudFromRosToOpen3d(latest_depth_image_msg)
        # pointcloud_open3d = pointcloud_open3d.random_down_sample(1/50.0)

        # input_tensor = torch.tensor(np.array(pointcloud_open3d.points),dtype=torch.float32)
        # manual_mask = torch.ones(input_tensor.shape[0],dtype=torch.float32)

        # ros_dataset = tgd.Data(
        #     id="box",
        #     pos=input_tensor,
        #     flow=torch.zeros_like(input_tensor,dtype=torch.float32),
        #     mask=manual_mask
        # )
        # ros_data= typing.cast(Flowbot3DTGData, ros_dataset)

        # batch = tgd.Batch.from_data_list([ros_data])
        # with torch.no_grad():
        #     pred_flow = self.model(batch.cuda()).cpu()

        # pred_flow_normalized = (pred_flow / pred_flow.norm(dim=1).max()).numpy()

        # line_list = Marker()
        # line_list.header.frame_id = pointcloud_msg.header.frame_id
        # line_list.header.stamp = pointcloud_msg.header.stamp
        # line_list.ns = "line_list"
        # line_list.action = Marker.ADD
        # line_list.pose.orientation.w = 1.0

        # line_list.id = 2
        # line_list.type = Marker.LINE_LIST
        # line_list.scale.x = 0.1

        # line_list.color.r = 1.0
        # line_list.color.a = 1.0

        # print(input_tensor[0])
        # print(pred_flow[0])

        # initial_points = convertTensorToPointsArray(input_tensor)
        # end_points = convertTensorToPointsArray(input_tensor + pred_flow)
        # interleave_list = list(itertools.chain(*zip(initial_points,end_points)))

        # print(initial_points[0])

        # print(end_points[0])
        # print(interleave_list[0])
        # print(interleave_list[1])

        # line_list.points = interleave_list

        # self.flow_pub.publish(line_list)

        # exit(0)

    def depthImageCallback(self, depth_img):

        self.depth_image_queue.append(depth_img)
        if (len(self.depth_image_queue) > self.queue_size):
            self.depth_image_queue.pop()


if __name__ == '__main__':

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument(
        '--model_path',
        default='/home/russell/git/flowbot3d/checkpoints/no-wandb/2023_05_10-11_29_09/epoch=99-step=78600.ckpt',
        type=str)

    args, unknown = parser.parse_known_args()

    rospy.init_node('flowbot3d_ros')

    rosFlowbot = RosFlowbot3d(args)

    rospy.spin()
