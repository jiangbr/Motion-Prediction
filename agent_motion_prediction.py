#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from typing import Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./agent_motion_config.yaml")

# model path and saving path
log_dir = '../input/'


# Our baseline is a simple `resnet50` pretrained on `imagenet`.
def build_model(cfg_: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model_ = resnet50(pretrained=False)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg_["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    model_.conv1 = nn.Conv2d(
        num_in_channels,
        model_.conv1.out_channels,
        kernel_size=model_.conv1.kernel_size,
        stride=model_.conv1.stride,
        padding=model_.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg_["model_params"]["future_num_frames"]
    model_.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model_


def forward(data_, model_, device_, criterion_):
    inputs = data_["image"].cuda(device_)
    target_availabilities = data_["target_availabilities"].unsqueeze(-1).cuda(device_)
    targets = data_["target_positions"].cuda(device_)
    # Forward pass
    outputs = model_(inputs).reshape(targets.shape)
    loss_ = criterion_(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss_ = loss_ * target_availabilities
    loss_ = loss_.mean()
    return loss_, outputs




# Training
def train(gpu, args):
    print('gpu{}: Begin training'.format(gpu))
    # distributed training initialization
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12121',
        world_size=args.world_size,
        rank=gpu
    )
    torch.manual_seed(0)

    # INIT MODEL
    model = build_model(cfg)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print("gpu{}: Finish constructing model".format(gpu))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    # Load the Train Data
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    print('gpu{}: Finish loading dataset'.format(gpu))

    # wrap the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                    rank=gpu, shuffle=train_cfg["shuffle"])
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_cfg["batch_size"],
                                  num_workers=0, pin_memory=True, sampler=train_sampler)

    # wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # using apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)
    # model = DDP(model)

    # loading pretrain model
    if cfg['train_params']['model_num'] != 0:
        model_path = log_dir + 'resnet_{}.pth'.format(cfg['train_params']['model_num'])
        model.load_state_dict(torch.load(model_path, map_location={'cuda:%d' % 0: 'cuda:%d' % gpu})['model'])
        print("gpu{}: Finish loading model".format(gpu))
        dist.barrier()
    else:
        print("gpu{}: No pretrain model".format(gpu))

    # TRAIN LOOP
    checkpoint = cfg['train_params']['checkpoint_every_n_steps']
    for epoch in range(cfg['train_params']['max_num_epochs']):
        print('gpu{}: Begin epoch {}'.format(gpu, epoch))
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(len(train_dataloader)))
        losses_train = []
        for index in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            loss, _ = forward(data, model, gpu, criterion)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

            if index % checkpoint == 0 and index != 0 and gpu == 0:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, log_dir + 'resnet_{}_{}.pth'.format(epoch, index))
    print("gpu{}: Finish training")

# evaluation
def eval_dataset():
    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]
    print(eval_cfg)
    print("开始加载eval数据")
    eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
                                            cfg["raster_params"]["filter_agents_threshold"],
                                            num_frames_to_chop, cfg["model_params"]["future_num_frames"],
                                            MIN_FUTURE_STEPS)
# "./evaluation_pred.csv""gd2000new.csv"
def eval_csv(predcsv, evalcsv):
    eval_base_path = os.environ["L5KIT_DATA_FOLDER"] + "/scenes/validate_chopped_100"
    eval_gt_path = str(Path(eval_base_path) / evalcsv)
    pred_path = predcsv
    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
        with open('likelihood', 'a') as file_handle:
            file_handle.write("{}:{}\n".format(metric_name, metric_mean))
def eval(gpu, args):
    print('gpu{}: Begin to evaluate'.format(gpu))
    # distributed training initialization
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12121',
        world_size=args.world_size,
        rank=gpu
    )
    torch.manual_seed(0)

    # INIT MODEL
    model = build_model(cfg)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print("gpu{}: Finish constructing model".format(gpu))
    criterion = nn.MSELoss(reduction="none")

    # wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # load trained model
    model_path = log_dir + 'resnet_1_{}.pth'.format(cfg['val_params']['model_num'])
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location={'cuda:%d' % 0: 'cuda:%d' % gpu})['model'])
    print("gpu{}: Finish loading model".format(gpu))
    dist.barrier()

    num_frames_to_chop = 100
    eval_cfg = cfg["val_data_loader"]
    print(eval_cfg)
    print("开始加载eval数据")
    # eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
    #                                         cfg["raster_params"]["filter_agents_threshold"],
    #                                         num_frames_to_chop, cfg["model_params"]["future_num_frames"],
    #                                         MIN_FUTURE_STEPS)
    eval_base_path = os.environ["L5KIT_DATA_FOLDER"] + "/scenes/validate_chopped_100"
    # eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
    eval_zarr_path = str(Path(eval_base_path) / "validate.zarr")
    eval_mask_path = str(Path(eval_base_path) / "mask.npz")
    eval_gt_path = str(Path(eval_base_path) / "gt.csv")

    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    eval_mask = np.load(eval_mask_path)["arr_0"]
    # 生成网格
    rasterizer = build_rasterizer(cfg, dm)
    # ===== INIT DATASET AND LOAD MASK
    eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
    # wrap the dataset
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=args.world_size,
                                                                   rank=gpu, shuffle=eval_cfg["shuffle"])
    eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
                                 num_workers=0, sampler=eval_sampler)
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []
    # i = 0
    print("len", len(eval_dataloader))
    progress_bar = tqdm(eval_dataloader)
    for data in progress_bar:
        # if i > 99:
        #     break
        _, ouputs = forward(data, model, gpu, criterion)
        future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
        # i = i+1
    pred_path = "./evaluation_pred_full.csv"

    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd),
                   )

    metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood])
    for metric_name, metric_mean in metrics.items():
        print(metric_name, metric_mean)
        with open('likelihood', 'a') as file_handle:
            file_handle.write("{}:{}\n".format(metric_name, metric_mean))

# Test
def test(gpu, args):
    print('gpu{}: Begin to predict'.format(gpu))
    # distributed training initialization
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12121',
        world_size=args.world_size,
        rank=gpu
    )
    torch.manual_seed(0)

    # INIT MODEL
    model = build_model(cfg)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print("gpu{}: Finish constructing model".format(gpu))
    criterion = nn.MSELoss(reduction="none")

    # Load the test Data
    test_cfg = cfg['test_data_loader']
    # 获取mask
    mask_path = os.environ["L5KIT_DATA_FOLDER"] + test_cfg["mask"]
    # 导入mask
    mask = np.load(mask_path)["arr_0"]
    # 生成网格
    rasterizer = build_rasterizer(cfg, dm)
    # 获取包含训练数据集的实例，内部包含agents、frames和scenes
    test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
    # 加入mask，仅对mask指出的71122个agent进行运动预测，这些agent均只有前100帧而没有未来的50帧
    test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=mask)

    # wrap the dataset
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.world_size,
                                                                   rank=gpu, shuffle=test_cfg["shuffle"])
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_cfg["batch_size"],
                                 num_workers=0, pin_memory=True, sampler=test_sampler)
    print(len(test_dataloader))
    # wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # load trained model
    model_path = log_dir + 'resnet_1_{}.pth'.format(cfg['test_params']['model_num'])
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['model'])
    # print("test所用模型加载成功")
    #
    # model_path = log_dir + 'resnet_{}.pth'.format(cfg['train_params']['model_num'])
    model.load_state_dict(torch.load(model_path, map_location={'cuda:%d' % 0: 'cuda:%d' % gpu})['model'])
    print("gpu{}: Finish loading model".format(gpu))
    dist.barrier()

    # # ===== GENERATE AND LOAD CHOPPED DATASET
    # eval_cfg = cfg['val_data_loader']
    # # 获取mask
    # mask_path = os.environ["L5KIT_DATA_FOLDER"] + eval_cfg["mask"]
    # # 导入mask
    # mask = np.load(mask_path)["arr_0"]
    # # 生成网格
    # rasterizer = build_rasterizer(cfg, dm)
    # # 获取包含训练数据集的实例，内部包含agents、frames和scenes
    # test_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
    # # 获取数据集内的特定内容
    # agent_dataset = AgentDataset(cfg, test_zarr, rasterizer)
    # # 加入mask，仅对mask指出的71122个agent进行运动预测，这些agent均只有前100帧而没有未来的50帧
    # mask_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=mask)
    #
    # eval_dataloader = DataLoader(mask_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"],
    #                              num_workers=eval_cfg["num_workers"])

    # EVAL LOOP
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []

    agent_ids = []
    progress_bar = tqdm(test_dataloader)
    for data in progress_bar:
        _, ouputs = forward(data, model, gpu, criterion)
        future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

    # Save results
    pred_path = "./pred_{}_gpu{}.csv".format(cfg['test_params']['model_num'], gpu)
    write_pred_csv(pred_path,
                   timestamps=np.concatenate(timestamps),
                   track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd),
                   )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    if not cfg["train_params"]["load_the_state"]:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    if cfg["test_params"]["load_the_state"]:
        args.gpus = 1
        args.world_size = 1
        mp.spawn(test, nprocs=args.gpus, args=(args,))
    if cfg["val_params"]["load_the_state"]:
        args.gpus = 1
        args.world_size = 1
        mp.spawn(eval, nprocs=args.gpus, args=(args,))
    if False:
        eval_dataset()
    if False:
        eval_csv(predcsv="./evaluation_pred.csv", evalcsv="gd2000new.csv")







if __name__ == '__main__':
    main()
