#!/usr/bin/env python

import argparse

import torch
import open3d
import numpy as np
from ctypes import * # convert float to uint32
import torch_geometric.data as tgd
import itertools
import typing


# ROS imports
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2

import ros_numpy

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
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
# https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
def convertCloudFromRosToOpen3d(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
             rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

        # rgb = 0.0*xyz

        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud


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
        
        # Publishers
        self.flow_pub = rospy.Publisher('articulation_flow', Marker, queue_size=10)
        # Subscribers
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.depthCameraCallback)
        # self.input_pointcloud_sub = message_filters.Subscriber("/camera/points", PointCloud2)
        # self.input_depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        # ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        # ts.registerCallback(self.depthCameraCallback)

    def depthCameraCallback(self, pointcloud_msg):

        print("Got depth and points")

        pointcloud_open3d = convertCloudFromRosToOpen3d(pointcloud_msg)
        pointcloud_open3d = pointcloud_open3d.random_down_sample(1/50.0)

        input_tensor = torch.tensor(np.array(pointcloud_open3d.points),dtype=torch.float32)
        manual_mask = torch.ones(input_tensor.shape[0],dtype=torch.float32)
        
        ros_dataset = tgd.Data(
            id="box",
            pos=input_tensor,
            flow=torch.zeros_like(input_tensor,dtype=torch.float32),
            mask=manual_mask
        )

        ros_data= typing.cast(Flowbot3DTGData, ros_dataset)

        batch = tgd.Batch.from_data_list([ros_data])
        with torch.no_grad():
            pred_flow = self.model(batch.cuda()).cpu()

        pred_flow_normalized = (pred_flow / pred_flow.norm(dim=1).max()).numpy()

        line_list = Marker()
        line_list.header.frame_id = pointcloud_msg.header.frame_id
        line_list.header.stamp = pointcloud_msg.header.stamp
        line_list.ns = "line_list"
        line_list.action = Marker.ADD
        line_list.pose.orientation.w = 1.0

        line_list.id = 2
        line_list.type = Marker.LINE_LIST
        line_list.scale.x = 0.1

        line_list.color.r = 1.0
        line_list.color.a = 1.0

        print(input_tensor[0])
        print(pred_flow[0])

        initial_points = convertTensorToPointsArray(input_tensor)
        end_points = convertTensorToPointsArray(input_tensor + pred_flow)
        interleave_list = list(itertools.chain(*zip(initial_points,end_points)))

        print(initial_points[0])
        
        print(end_points[0])
        print(interleave_list[0])
        print(interleave_list[1])

        line_list.points = interleave_list

        self.flow_pub.publish(line_list)

        exit(0)


if __name__ == '__main__':

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--model_path',
                        default='/home/russell/git/flowbot3d/checkpoints/no-wandb/no_mask/epoch=99-step=78600.ckpt',
                        type=str)

    args = parser.parse_args()

    rospy.init_node('imap_ros')

    # mp.set_start_method('spawn')  # Required for m/ultiprocessing on some platforms

    rosFlowbot = RosFlowbot3d(args)

    rospy.spin()