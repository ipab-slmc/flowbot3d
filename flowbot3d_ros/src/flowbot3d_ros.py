#!/usr/bin/env python

import argparse

import torch
import open3d
import numpy as np
from ctypes import *  # convert float to uint32
import torch_geometric.data as tgd
import itertools
import typing
from scipy.spatial.transform import Rotation
import gc

# ROS imports
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Point, PointStamped
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import tf2_ros
import cv2

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
FIELDS_I = [
    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
]
FIELDS_NORM = [
    PointField(name='normal_x', offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name='normal_y', offset=20, datatype=PointField.FLOAT32, count=1),
    PointField(name='normal_z', offset=24, datatype=PointField.FLOAT32, count=1),
    PointField(name='curvature', offset=28, datatype=PointField.FLOAT32, count=1),
]

FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

FIELDS_XYZINORM = FIELDS_XYZ + FIELDS_I + FIELDS_NORM

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
    points = open3d.core.Tensor.numpy(open3d_cloud.point["points"]).astype(np.float32)
    if len(open3d_cloud.point["colors"]) == 0:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(open3d.core.Tensor.numpy(open3d_cloud.point["colors"]))  # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        colors = colors.astype(np.uint32)
        points_list = points.tolist()
        colors_list = colors.reshape(-1, 1).tolist()
        cloud_data = list(map(list.__add__, points_list, colors_list))

    return pc2.create_cloud(header, fields, cloud_data)


def convertTensorToRosPointcloud(points_tensor, colors_tensor, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    fields = FIELDS_XYZINORM

    intensity_tensor = torch.zeros(points_tensor.shape[0],1)
    curvature_tensor = intensity_tensor

    cloud_data = torch.cat((points_tensor, intensity_tensor, colors_tensor, curvature_tensor), 1)

    cloud_data = cloud_data.tolist()

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

        self.load_model = rospy.get_param("~load_model")

        if(self.load_model):
            print("LOADING MODEL")
        else:
            print("NOT LOADING MODEL")

        self.depth_image_queue = []
        self.queue_size = 10
        self.bridge = CvBridge()

        # real camera
        self.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=613.7716064453125, fy=614.099609375, cx=314.3857421875, cy=242.46636962890625)

        # simulation
        # self.camera_intrinsics = open3d.camera.PinholeCameraIntrinsic(
            # width=640, height=480, fx=462.1379699707031, fy=462.1379699707031, cx=320.0, cy=240.0)

        self.camera_k_matrix = open3d.core.Tensor(self.camera_intrinsics.intrinsic_matrix)


        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.go_to_grasp_srv_ = rospy.ServiceProxy('/am_ros/move_to_grasp', Empty)

        self.rate = rospy.Rate(100)  # 100hz
        self.tf_msg = TransformStamped()
        self.got_click = False
        self.got_mask = False
        self.got_mask_once = False

        # Publishers
        self.pointcloud_pub = rospy.Publisher('~masked_pointcloud', PointCloud2, queue_size=10)
        self.flow_pub = rospy.Publisher('~affordance', PointCloud2, queue_size=10)
        self.flow_pub_viz = rospy.Publisher('~affordance_visualization', Marker, queue_size=10)
        # Subscribers  
        if (self.load_model):
            self.model = fmf.ArtFlowNet.load_from_checkpoint(args.model_path).cuda()
            self.model.eval()
            rospy.Subscriber("/input/mask", Image, self.maskedImageCallback, queue_size=10)

        rospy.Subscriber("/rqt_image_segmentation/click_point",
                         Point, self.clickPointCallback, queue_size=10)
        rospy.Subscriber("/input/depth_image",
                         Image, self.depthImageCallback, queue_size=10)

        self.click_goal_offest = [-0.02, 0.0, 0.0] # in world frame

    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def maskedImageCallback(self, image_mask_msg):
        # convert image mask to open3d
        image_mask_opencv = self.bridge.imgmsg_to_cv2(image_mask_msg, desired_encoding='32FC1')
        zero_channel = np.zeros_like(image_mask_opencv)
        image_mask_opencv_bgr = cv2.merge((image_mask_opencv, zero_channel, zero_channel))
        image_mask_open3d = open3d.t.geometry.Image(open3d.core.Tensor(image_mask_opencv_bgr))

        # convert depth image to open3d
        latest_depth_image_msg = self.depth_image_queue.pop()
        depth_img_opencv = self.bridge.imgmsg_to_cv2(latest_depth_image_msg, desired_encoding='32FC1')
        depth_img_open3d = open3d.t.geometry.Image(open3d.core.Tensor(depth_img_opencv))

        # create rgbd
        rgbd_image = open3d.t.geometry.RGBDImage(image_mask_open3d, depth_img_open3d)
        pointcloud_open3d = open3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_k_matrix, stride=10)

        point_cloud_msg = convertCloudFromOpen3dToRos(
            pointcloud_open3d, frame_id=latest_depth_image_msg.header.frame_id)
        self.pointcloud_pub.publish(point_cloud_msg)

        input_tensor = torch.utils.dlpack.from_dlpack(pointcloud_open3d.point["points"].to_dlpack()).type(torch.float32)
        masking_colors = torch.utils.dlpack.from_dlpack(pointcloud_open3d.point["colors"].to_dlpack()).type(torch.uint8)
        mask = torch.zeros(input_tensor.shape[0])

        i = 0
        for p in range(masking_colors.shape[0]):
            if (masking_colors[p][0] == 255 and masking_colors[p][1] == 0 and masking_colors[p][2] == 0):
                mask[i] = 1
            i += 1

        ros_dataset = tgd.Data(
            id="box",
            pos=input_tensor,
            flow=torch.zeros_like(input_tensor, dtype=torch.float32),
            mask=mask
        )
        ros_data = typing.cast(Flowbot3DTGData, ros_dataset)
        batch = tgd.Batch.from_data_list([ros_data])
        pred_flow = torch.zeros_like(input_tensor)
        with torch.no_grad():
            pred_flow = self.model(batch.cuda()).cpu()

        # Get masked points only
        mask_idx = mask == 1
        masked_input = input_tensor[mask_idx]
        masked_pred_flow = pred_flow[mask_idx]

        # convert to pointcloud for estimation
        output_cloud_msg = convertTensorToRosPointcloud(masked_input, masked_pred_flow, frame_id = point_cloud_msg.header.frame_id)
        self.flow_pub.publish(output_cloud_msg)

        line_list = Marker()
        line_list.header.frame_id = point_cloud_msg.header.frame_id
        line_list.header.stamp = point_cloud_msg.header.stamp
        line_list.ns = "line_list"
        line_list.action = Marker.ADD
        line_list.pose.orientation.w = 1.0

        line_list.id = 2
        line_list.type = Marker.LINE_LIST
        line_list.scale.x = 0.001

        line_list.color.r = 1.0
        line_list.color.a = 1.0


        flow_scale = 0.05
        initial_points = convertTensorToPointsArray(masked_input)
        end_points = convertTensorToPointsArray(masked_input + flow_scale*masked_pred_flow)
        interleave_list = list(itertools.chain(*zip(initial_points, end_points)))

        line_list.points = interleave_list
        self.flow_pub_viz.publish(line_list)

        self.got_mask = True
        self.got_mask_once = True

    def convertMsgToMatrix(self, trans):
        r = Rotation.from_quat([trans.transform.rotation.x, trans.transform.rotation.y,
                                trans.transform.rotation.z, trans.transform.rotation.w])

        mat = np.identity(4)
        mat[0:3, 0:3] = r.as_matrix()
        mat[0:3, 3] = np.array([trans.transform.translation.x, trans.transform.translation.y,
                               trans.transform.translation.z])

        return mat

    def clickPointCallback(self, click_point):
        # Project point into 3D
        latest_depth_image_msg = self.depth_image_queue.pop()
        depth_img_opencv = self.bridge.imgmsg_to_cv2(latest_depth_image_msg, desired_encoding='32FC1')

        point2d = torch.reshape(torch.Tensor([click_point.x, click_point.y, 1.0]), (3, 1))
        camera_matrix = torch.Tensor(self.camera_intrinsics.intrinsic_matrix)
        point3d = torch.matmul(camera_matrix.inverse(), point2d)

        depth = depth_img_opencv[int(click_point.y)][int(click_point.x)] * 0.001  # conver mm to m

        point3d_position = np.array([point3d[0][0] * depth, point3d[1][0] * depth, depth])
        point3d_transform = np.identity(4)
        point3d_transform[0:3, 3] = point3d_position

        try:
            T_BC = self.tfBuffer.lookup_transform(
                "base_link_base", latest_depth_image_msg.header.frame_id, rospy.Time(),
                timeout=rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"can't get transform from base_link_base to {latest_depth_image_msg.header.frame_id}")

        trans_in_base = self.convertMsgToMatrix(T_BC) @ point3d_transform

        self.tf_msg.header.frame_id = "base_link_base"
        self.tf_msg.child_frame_id = "grasp_goal"
        self.tf_msg.transform.translation.x = trans_in_base[0][3] + self.click_goal_offest[0]
        self.tf_msg.transform.translation.y = trans_in_base[1][3] + self.click_goal_offest[1]
        self.tf_msg.transform.translation.z = trans_in_base[2][3] + self.click_goal_offest[2]
        self.tf_msg.transform.rotation.x = 0
        self.tf_msg.transform.rotation.y = 0
        self.tf_msg.transform.rotation.z = 0
        self.tf_msg.transform.rotation.w = 1
        self.got_click = True
        
        self.tf_msg.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(self.tf_msg)


    def run(self):

        while not rospy.is_shutdown():
            if self.got_mask_once:

                self.tf_msg.header.stamp = rospy.Time.now()
                self.tf_broadcaster.sendTransform(self.tf_msg)

            self.rate.sleep()

    def depthImageCallback(self, depth_img):

        self.depth_image_queue.append(depth_img)
        if (len(self.depth_image_queue) > self.queue_size):
            self.depth_image_queue.pop()


if __name__ == '__main__':

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument(
        '--model_path',
        default='/home/russell/git/flowbot3d/checkpoints/no-wandb/camera_frame/mask/epoch=99-step=78600.ckpt',
        type=str)

    args, unknown = parser.parse_known_args()

    rospy.init_node('flowbot3d_ros')

    rosFlowbot = RosFlowbot3d(args)

    rosFlowbot.run()
