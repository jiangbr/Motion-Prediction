from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
import os
import numpy as np
from HandwriteDataset import HandwriteDataset
from l5kit.dataset import EgoDataset, AgentDataset

np.set_printoptions(threshold=np.inf)
os.environ["L5KIT_DATA_FOLDER"] = '../input/lyft-motion-prediction-autonomous-vehicles'
cfg = load_config_data('./config/l5kit_config.yaml')
dm = LocalDataManager()
dataset_path = dm.require(cfg["train_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
agents = zarr_dataset.agents
'''
mask_and_track_id = np.load("./data/mask&track_id.npz")
mask = np.load(os.environ["L5KIT_DATA_FOLDER"] + '/scenes/mask.npz')
agents_mask = mask_and_track_id["agents_mask"]
agents_track_id_in_frames = mask_and_track_id["agents_track_id"]
mask_agent = mask['arr_0']
'''
rast = build_rasterizer(cfg, dm)
dataset = HandwriteDataset(cfg,zarr_dataset,rast,"./data/mask&track_id.npz")
data = dataset[1000]
print(data["select_track_id"])
