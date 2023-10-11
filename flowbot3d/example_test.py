# %load_ext autoreload
# %autoreload 2

import open3d as o3d
import numpy as np
import time
from typing import cast

import os
import pandas as pd
from pathlib import Path
from IPython.display import display

import torch
import torch_geometric.data as tgd
import rpad.partnet_mobility_utils.dataset as rpd


from flowbot3d.datasets.flow_dataset_pyg import Flowbot3DPyGDataset, Flowbot3DTGData
# import flowbot3d.tg_dataset as ftd
import flowbot3d.models.artflownet as fmf



def make_tables(results_dir, method_names, metric, method_display_names=None, higherbetter=True):
    dfs = []
    if method_display_names is None:
        method_display_names = method_names
    for split in ['umpnet-train-train', 'umpnet-train-test', 'umpnet-test']:
        rows = []
        for method_name in method_names:
            csv = Path(results_dir) / method_name / f"{split}.csv"
            df = pd.read_csv(csv, index_col=0)
            rows.append(df.loc[metric])
        df = pd.DataFrame(rows, method_display_names)
        df = df.style.set_caption(split)
        if higherbetter:
            df = df.highlight_max(color="lightblue")
        else:
            df = df.highlight_min(color="lightblue")
        dfs.append(df)
    return dfs
        

# Create the dataset.
print("loading dataset")
dataset = Flowbot3DPyGDataset(
    root=os.path.expanduser("/home/russell/Datasets/partnet-mobility-v0/dataset/raw/"),
    split=["104046"],
    randomize_camera=False,
)

print("Load a ply point cloud, print it, and render it")

# Load the model.
print("loading checkpoint")
ckpt_path = "/home/russell/git/flowbot3d/checkpoints/no-wandb/2023_10_11-11_21_49/epoch=99-step=200.ckpt"
model = fmf.ArtFlowNet.load_from_checkpoint(ckpt_path).cuda()
model.eval()

################ Read file
# pcd = o3d.t.io.read_point_cloud("/home/russell/test.pcd")
# input_tensor = torch.utils.dlpack.from_dlpack(pcd.point.positions.to_dlpack())
# mask = torch.zeros(pcd.point.positions.shape[0])
# i = 0
# for point in pcd.point.colors:
#     if not( point[0] == 0 and point[1] == 0 and point[2] == 0):
#         mask[i] = 1
#     i += 1

# ros_dataset = tgd.Data(
#     id="box",
#     pos=input_tensor,
#     flow=torch.zeros_like(input_tensor,dtype=torch.float32),
#     mask=mask
# )
# ros_dataset= cast(Flowbot3DTGData, ros_dataset)

print("here I guess")
time1 = time.perf_counter()
data = dataset.get_data("104046")
batch = tgd.Batch.from_data_list([data])
print("forward pass")
with torch.no_grad():
    pred_flow = model(batch.cuda()).cpu()

print(f"Took {time.perf_counter() - time1} seconds")
# Display the figure.
input = batch.cpu()
input.flow = pred_flow

# mask_idx = mask == 1

print(f"pred_flow {pred_flow}")
# print(f"masked_pred_flow {pred_flow[mask_idx]}")

fig = fmf.ArtFlowNet.make_plots(pred_flow, input)["artflownet_plot"]

fig.update_layout(autosize=False, height=800, width=1200)

fig.show()
