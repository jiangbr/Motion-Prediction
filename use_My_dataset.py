from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.dataset import AgentDataset
import os
from MyDataset import MyDataset
from torch.utils.data import DataLoader
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
cfg = load_config_data('./config/l5kit_config.yaml')
dm = LocalDataManager(None)
train_cfg = cfg["train_data_loader"]
dataset_path = dm.require(train_cfg["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()

rast = build_rasterizer(cfg, dm)
my_dataset = MyDataset(cfg, zarr_dataset, rast)
agent_dataset = AgentDataset(cfg, zarr_dataset, rast)

my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=0)
agent_dataloader = DataLoader(agent_dataset, batch_size=1, shuffle=False, num_workers=0)



data = my_dataset[1024]
print(data["neighbor_agents_history_position"])
print(data["neighbor_agents_history_position"].shape)
print(data["My_agent_history_position"])
print(data["My_agent_history_position"].shape)
print(data["neighbor_agents_track_id"])
print(data["My_agent_track_id"])
