# -*- coding: utf-8 -*-

import gin
import numpy as np
from train import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader


def save(rank):
    gin.parse_config_file('./config/config.gin')
    params = Params()()

    # set env variable and get config
    os.environ["L5KIT_DATA_FOLDER"] = params.env
    dm = LocalDataManager(None)
    cfg = load_config_data(params.l5kit)

    # Load the dataset
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    prior_info_path = train_cfg["prior_info"]
    hand_dataset = HandwriteDataset(cfg, train_zarr, rasterizer, prior_info_path)

    # wrap dataset with StampDataset
    train_dataset = StampDataset(hand_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=12,
                                                                    rank=rank, shuffle=False)
    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=train_dataset.collate_fn,
                                   num_workers=8, pin_memory=True, sampler=train_sampler)

    # save all the data
    hist = []
    nbr = []
    fut = []
    mask = []
    nbr_info = []
    index = []
    length = len(train_data_loader)
    for i, data in enumerate(train_data_loader):
        data = train_dataset[i]
        hist.append(data[0].flatten())
        nbr.append(data[1].flatten())
        fut.append(data[2].flatten())
        mask.append(data[3].flatten())
        nbr_info.append(data[5])
        index.append(data[6])

        if i % 100 == 99:
            print('Finish saving index {} of {}'.format(i, length))
    hist = np.asarray(hist, dtype=object)
    nbr = np.asarray(nbr, dtype=object)
    fut = np.asarray(fut, dtype=object)
    mask = np.asarray(mask, dtype=object)
    nbr_info = np.asarray(nbr_info, dtype=object)
    index = np.asarray(index, dtype=object)

    np.savez('./data/test.npz', hist=hist, nbr=nbr, fut=fut, mask=mask, nbr_info=nbr_info, index=index)


def load():
    data = np.load('./data/test.npz', allow_pickle=True)
    hist = data['hist'][1].reshape(100, -1, 2)
    nbr = data['nbr'][1].reshape(100, -1, 2)
    fut = data['fut'][1].reshape(50, -1, 2)
    mask = data['mask'][1].reshape(50, -1, 1)
    nbr_info = data['nbr_info'][1][0]
    index = data['index'][1]
    pass


if __name__ == '__main__':
    mp.spawn(save, nprocs=12)
    # load()
