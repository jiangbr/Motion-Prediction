#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

import gin
import numpy as np
import pandas as pd
from typing import *
from attrdict import AttrDict

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

from model_mfp import mfpNet
from my_utils import *
from HandwriteDataset import HandwriteDataset


def setup_logger(root_dir: str, rank: int) -> Tuple[Any, Any]:
    """Setup the data logger for logging"""
    import time
    import datetime
    import os

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y.%m.%d_%H.%M.%S')
    logging_dir = root_dir + "%s/" % timestamp
    if not os.path.isdir(logging_dir) and rank == 0:
        os.makedirs(logging_dir)
        os.makedirs(logging_dir + '/checkpoints')
        print("GPU{}: '".format(rank) + logging_dir + "' CREATED")
    dist.barrier()

    logger_file = open(logging_dir + '/logging_{}.log'.format(rank), 'w')
    return logger_file, logging_dir


@gin.configurable
class Params(object):
    def __init__(self, log: bool = False,  # save checkpoints?
                 modes: int = 3,  # how many latent modes
                 encoder_size: int = 16,  # encoder latent layer size
                 decoder_size: int = 16,  # decoder latent layer size
                 subsampling: int = 2,  # factor subsample in time
                 hist_len_orig_hz: int = 30,  # number of original history samples
                 fut_len_orig_hz: int = 50,  # number of original future samples
                 dyn_embedding_size: int = 32,  # dynamic embedding size
                 input_embedding_size: int = 32,  # input embedding size
                 dec_nbr_enc_size: int = 8,  # decoder neighbors encode size
                 nbr_atten_embedding_size: int = 80,  # neighborhood attention embedding size
                 seed: int = 1234,
                 remove_y_mean: bool = False,  # normalize by remove mean of the future trajectory
                 use_gru: bool = True,  # GRUs instead of LSTMs
                 bi_direc: bool = False,  # bidrectional
                 self_norm: bool = False,  # normalize with respect to the current time
                 data_aug: bool = False,  # data augment
                 use_context: bool = False,  # use contextual image as additional input
                 nll: bool = True,  # negative log-likelihood loss
                 use_forcing: int = 0,  # teacher forcing
                 iter_per_err: int = 100,  # iterations to display errors
                 iter_per_eval: int = 1000,  # iterations to eval on validation set
                 training_mode_update: int = 20000,  # switch training forcing
                 pre_train_num_updates: int = 200000,  # how many iterations for pretraining
                 updates_div_by_10: int = 100000,  # at what iteration to divide the learning rate by 10.0
                 nbr_search_depth: int = 10,  # how deep do we search for neighbors
                 lr_init: float = 0.001,  # initial learning rate
                 min_lr: float = 0.00005,  # minimal learning rate
                 use_cuda: bool = True,
                 iters_per_save: int = 1500,
                 epoch: int = 10,
                 env: str = '',
                 l5kit: str = '') -> None:
        # function locals() returns local dictionary
        self.params = AttrDict(locals())

    def __call__(self) -> Any:
        return self.params


class StampDataset(Dataset):
    def __init__(self, my_dataset: HandwriteDataset, gt_path: Optional[str] = None):
        """The dataset is used to transform items of MyDataset into desired format"""
        self.max_len = 0
        self.length = len(my_dataset)
        # define item list and assign transform
        self.dataset = my_dataset
        # define ground truth
        self.gt = None
        if gt_path is not None:
            self.gt = pd.read_csv(gt_path)
            # define num_index
            self.num_index = my_dataset.num_index

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """Return corresponding item"""
        data = self.dataset[index]
        # extract history trajectory
        hist_origin = data['history_agents_position']
        # extract future trajectory
        fut_origin = data['future_agents_position']
        # extract status
        status = data['agents_available_frames']
        # extract target index
        target_index = data['select_track_id']

        # filter out invalid training data
        hist = []
        fut = []
        masks = []
        hist_masks = []
        new_index = []
        # define minimum history and future trajectory
        min_hist = 50
        if self.gt is None:
            min_fut = 25
        else:
            min_fut = 0
        # traverse through the whole dataset
        for i in range(hist_origin.shape[0]):
            # extract agents appear in current timestamp
            if status[i, 2] == 1:
                # extract agents without label "None" or agents moving dramatically
                if status[i, 3] == 1 or status[i, 4] == 1:
                    # extract agents with sufficient data
                    if (status[i, 0] >= min_hist and status[i, 1] >= min_fut) \
                            or i in target_index or (status[i, 4] == 1 and status[i, 1] >= 1):
                        hist.append(hist_origin[i])
                        # extract absolute position
                        pos = hist_origin[i][-1, :].reshape(1, 2)
                        fut.append(fut_origin[i] - pos)
                        if i in target_index:
                            new_index.append(len(hist) - 1)
                        # define mask
                        mask = np.zeros([fut_origin.shape[1], 1])
                        mask[0:int(status[i, 1]), :] = 1
                        masks.append(mask)
                        # define history mask
                        hist_mask = np.zeros([hist_origin.shape[1], 1])
                        hist_mask[-int(status[i, 0]):, :] = 1
                        hist_masks.append(hist_mask)
        hist = np.stack(hist, axis=1)
        fut = np.stack(fut, axis=1)
        masks = np.stack(masks, axis=1)
        hist_masks = np.stack(hist_masks, axis=1)
        new_index = np.array(new_index)
        assert new_index.shape[0] == target_index.shape[0], 'index not matching'

        # define context
        context = None

        # we choose neighbor agents with a limitation, so as to avoid too much space assumption
        if hist.shape[1] <= 30:
            # define neighbors to be all the rest agents
            nbrs = [np.delete(hist, i, axis=1) for i in range(hist.shape[1])]
            nbrs = np.concatenate(nbrs, axis=1)
            # define masks
            nbrs_mask = [np.delete(hist_masks, i, axis=1) for i in range(hist_masks.shape[1])]
            nbrs_mask = np.concatenate(nbrs_mask, axis=1)

            # define nbrs_info
            index_list = np.array(range(hist.shape[1]))
            neighbor_list = [np.delete(index_list, i) for i in range(hist.shape[1])]
            nbrs_info = [{i: nbr for i, nbr in enumerate(neighbor_list)}]
        else:
            # limit neighbors in an certain region
            ref_pos = hist[-1, :, :]
            nbrs = []
            nbrs_mask = []
            neighbor_list = []
            # search neighbors for all the agents
            for i in range(hist.shape[1]):
                # define candidate neighbors
                candidate = np.delete(hist, i, axis=1)
                candidate_mask = np.delete(hist_masks, i, axis=1)
                candidate_index = np.delete(np.array(range(hist.shape[1])), i)
                # calculate distance
                distance = candidate[-1, :, :] - ref_pos[i:i + 1, :]
                distance = np.sqrt(np.sum(np.square(distance), axis=1))
                aver_distance = np.average(distance)
                # we choose threshold to be 50
                nbrs_temp = []
                list_temp = []
                mask_temp = []
                for j, data in enumerate(distance):
                    if data <= aver_distance:
                        nbrs_temp.append(candidate[:, j:j + 1, :])
                        mask_temp.append(candidate_mask[:, j:j + 1, :])
                        list_temp.append(candidate_index[j])
                nbrs_temp = np.concatenate(nbrs_temp, axis=1)
                mask_temp = np.concatenate(mask_temp, axis=1)
                list_temp = np.array(list_temp)
                nbrs.append(nbrs_temp)
                nbrs_mask.append(mask_temp)
                neighbor_list.append(list_temp)
            # define nbrs and nbrs_info
            nbrs = np.concatenate(nbrs, axis=1)
            nbrs_mask = np.concatenate(nbrs_mask, axis=1)
            nbrs_info = [{i: nbr for i, nbr in enumerate(neighbor_list)}]

        # define future trajectory for exact agents
        if self.gt is not None:
            # extract fut and mask from csv file
            region = slice(self.num_index[index], self.num_index[index + 1] - 1)
            csv_item = np.array(self.gt.loc[region])
            assert csv_item.shape[0] == new_index.shape[0], 'Error: Not matching'
            # define mask
            masks = csv_item[:, 2:52].reshape(-1, 50, 1).transpose(1, 0, 2)
            # define x coordinate
            x_axis = csv_item[:, 52:151:2].reshape(-1, 50, 1).transpose(1, 0, 2)
            # define y coordinate
            y_axis = csv_item[:, 53:152:2].reshape(-1, 50, 1).transpose(1, 0, 2)
            # define fut
            fut = np.concatenate([x_axis, y_axis], axis=2)
        # combine to be a tuple
        item = (hist, nbrs, fut, masks, hist_masks, nbrs_mask, context, nbrs_info, new_index)

        '''
        if nbrs.shape[1] > self.max_len:
            self.max_len = nbrs.shape[1]
            print("Index {}: Max len {}".format(index, self.max_len))
        '''
        return item

    @staticmethod
    def collate_fn(samples: List[Any]) -> Tuple:
        """Return desired format for DataLoader"""
        hist, nbrs, fut, mask, hist_mask, nbrs_mask, context, nbrs_info, index = samples[0]
        # transform to tensor
        hist = torch.from_numpy(hist).float()
        nbrs = torch.from_numpy(nbrs).float()
        fut = torch.from_numpy(fut).float()
        mask = torch.from_numpy(mask).float()
        hist_mask = torch.from_numpy(hist_mask).float()
        nbrs_mask = torch.from_numpy(nbrs_mask).float()
        if context is not None:
            context = torch.from_numpy(context).float()

        return hist, nbrs, fut, mask, hist_mask, nbrs_mask, context, nbrs_info, index


#####################################################################################################


def train(rank: int, args: Any) -> None:
    """Main training function"""

    # distributed training initialization
    gpu = args.list[rank]
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.gpus,
        rank=rank
    )
    print('Begin training with GPU{}'.format(gpu))

    # import config with gin
    gin.parse_config_file(args.config)
    params = Params()()

    # assign random seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

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

    # wrap the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus,
                                                                    rank=rank, shuffle=train_cfg["shuffle"])
    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=train_dataset.collate_fn,
                                   num_workers=0, pin_memory=True, sampler=train_sampler)
    print('GPU{}: Size of training dataset is {}'.format(gpu, len(train_data_loader)))

    # define path of evaluate dataset
    eval_cfg = cfg["val_data_loader"]
    eval_zarr_path = dm.require(eval_cfg["key"])
    eval_mask_path = eval_cfg["mask"]
    eval_gt_path = dm.require(eval_cfg["gt"])

    # import evaluate dataset
    eval_zarr = ChunkedDataset(eval_zarr_path).open()
    hand_dataset = HandwriteDataset(cfg, eval_zarr, rasterizer, agents_mask_path=eval_mask_path)
    eval_dataset = StampDataset(hand_dataset, eval_gt_path)

    # wrap evaluate dataset
    eval_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus,
                                                                   rank=rank, shuffle=eval_cfg["shuffle"])
    eval_data_loader = DataLoader(eval_dataset, shuffle=False, batch_size=1, collate_fn=eval_dataset.collate_fn,
                                  num_workers=0, pin_memory=True, sampler=eval_sampler)

    # Initialize network
    net = mfpNet(params)
    torch.cuda.set_device(gpu)
    net.cuda(gpu)

    # wrap model
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    # load pre-train model
    if train_cfg['model'] != "":
        save_cuda = train_cfg['cuda']
        net.load_state_dict(torch.load(train_cfg['model'], map_location={'cuda:%d' % save_cuda: 'cuda:%d' % gpu}))
        print('GPU{}: Finish importing pre-train model'.format(gpu))
        dist.barrier()

    # define logging file
    logger_file, logging_dir = None, None
    if params.log:
        logger_file, logging_dir = setup_logger("./checkpts/", rank)

    train_loss: List = []

    # For efficiency, we first pre-train w/o interactive rollouts
    MODE = 'EndPre'
    use_forcing = params.use_forcing
    num_updates = 0
    optimizer = None

    for epoch_num in range(params.epoch):
        if MODE == 'EndPre':
            MODE = 'Train'
            print('GPU{}: Training with interactive rollouts.'.format(gpu))
            bStepByStep = True
        else:
            print('GPU{}: Pre-training without interactive rollouts.'.format(gpu))
            bStepByStep = False

        # Average losses.
        avg_tr_loss = []

        # begin training
        for data in train_data_loader:
            # transform from Pre mode to EndPre mode
            if num_updates > params.pre_train_num_updates and MODE == 'Pre':
                MODE = 'EndPre'
                break

            # switch from teacher forcing to classmate forcing further to no forcing
            if num_updates % params.training_mode_update == params.training_mode_update - 1:
                if use_forcing == 1:
                    use_forcing = 2
                elif use_forcing == 2:
                    use_forcing = 0

            # Determine learning rate
            lr_fac = np.power(0.2, num_updates // params.updates_div_by_10)
            lr = max(params.min_lr, params.lr_init * lr_fac)
            if optimizer is None:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            elif lr != optimizer.defaults['lr']:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            # extract data from dataset
            # hist is [seq_len, num_agents, 2]
            # nbrs is [seq_len, num_nbrs, 2]
            # fut is [fut_len, num_agents, 2]
            # mask is [fut_len, num_agents, 1]
            # context is [num_agents, 3, 96, 320]
            # nbrs_info[0] is dictionary with {0: nbrs, 1:nbrs}
            hist, nbrs, fut, mask, hist_mask, nbrs_mask, context, nbrs_info, index = data
            hist = hist.cuda(gpu)
            nbrs = nbrs.cuda(gpu)
            fut = fut.cuda(gpu)
            mask = mask.cuda(gpu)
            hist_mask = hist_mask.cuda(gpu)
            nbrs_mask = nbrs_mask.cuda(gpu)
            if context is not None:
                context = context.cuda(gpu)

            # Forward pass.
            fut_preds, modes_pred = net.module.forward_mfp(hist, nbrs, mask, hist_mask, nbrs_mask, context, nbrs_info,
                                                           fut, bStepByStep, use_forcing=use_forcing)
            # transform predict to [modes, fut_len, num_agents, 2]
            fut_preds = torch.stack(fut_preds, 0)[:, :, :, :2]

            # calculate loss
            l = nll_loss_multimodes(fut_preds, fut, mask, modes_pred)
            dist.barrier()

            # Backpropagation.
            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()
            num_updates += 1

            avg_tr_loss.append(l.item())

            # effective_batch_sz = float(hist.shape[1])
            if num_updates % params.iter_per_err == params.iter_per_err - 1:
                train_loss.append(np.mean(avg_tr_loss))
                print("GPU{}: Epoch no:".format(rank), epoch_num, "update:", num_updates, "| Avg train loss:",
                      format(np.mean(avg_tr_loss), '0.4f'), " learning_rate:%.5f" % lr)

                if params.log:
                    msg_str_ = ("GPU{}: Epoch no:".format(rank), epoch_num, "update:", num_updates, "| Avg train loss:",
                                format(np.mean(avg_tr_loss), '0.4f'), " learning_rate:%.5f" % lr)
                    msg_str = str([str(ss) for ss in msg_str_])
                    logger_file.write(msg_str + '\n')
                    logger_file.flush()

                avg_tr_loss = []

            if num_updates % params.iter_per_eval == params.iter_per_eval - 1:
                print("GPU{}: Starting eval".format(rank))
                val_nll_err = evaluate(net, params, eval_data_loader, bStepByStep,
                                       use_forcing=0, num_batches=500)
                print("GPU{}: val nll= {}".format(rank, val_nll_err))
                if params.log:
                    logger_file.write('val nll: ' + str(val_nll_err) + '\n')
                    logger_file.flush()

            # Save weights.
            if params.log and num_updates % params.iters_per_save == params.iters_per_save - 1 and rank == 0:
                msg_str = '\nSaving state, update iter:%d %s' % (num_updates, logging_dir)
                print(msg_str)
                logger_file.write(msg_str)
                logger_file.flush()
                torch.save(net.state_dict(),
                           logging_dir + '/checkpoints/ngsim_%06d' % num_updates + '.pth')  # type: ignore


def evaluate(net: torch.nn.Module, params: AttrDict, data_loader: DataLoader, bStepByStep: bool,
             use_forcing: int, num_batches: int) -> float:
    """Evaluation function for validation and test data.
    Given a MFP network, data loader, evaluate either NLL error.
    """
    with torch.no_grad():
        lossVals = 0
        counts = 0

        for i, data in enumerate(data_loader):
            # iter for certain times
            if i >= num_batches:
                break
            hist, nbrs, fut, mask, hist_mask, nbrs_mask, context, nbrs_info, index = data
            if params.use_cuda:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                fut = fut.cuda()
                mask = mask.cuda()
                hist_mask = hist_mask.cuda()
                nbrs_mask = nbrs_mask.cuda()
                if context is not None:
                    context = context.cuda()

            # Forward pass
            fut_preds, modes_pred = net.module.forward_mfp(hist, nbrs, mask, hist_mask, nbrs_mask, context, nbrs_info,
                                                           fut, bStepByStep, use_forcing=use_forcing)
            # transform predict to [modes, fut_len, num_agents, 2]
            fut_preds = torch.stack(fut_preds, 0)[:, :, :, :2]

            # calculate loss
            l = nll_loss_multimodes(fut_preds, fut, mask, modes_pred, index=index)

            lossVals += l.detach().cpu()
            counts += index.shape[0]

        err = lossVals / counts
    return err


#####################################################################################################


def test(rank: int, args: Any) -> None:
    """Generate final result from test.zarr"""
    # distributed testing initialization in order to import pretrain model
    gpu = args.list[rank]
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12222',
        world_size=1,
        rank=rank
    )
    print('Begin testing with GPU{}'.format(gpu))

    # import config with gin
    gin.parse_config_file(args.config)
    params = Params()()

    # set env variable and get config
    os.environ["L5KIT_DATA_FOLDER"] = params.env
    dm = LocalDataManager(None)
    cfg = load_config_data(params.l5kit)
    rasterizer = build_rasterizer(cfg, dm)

    # define path of evaluate dataset
    test_cfg = cfg["val_data_loader"]
    test_zarr_path = dm.require(test_cfg["key"])
    test_mask_path = test_cfg["mask"]

    # import evaluate dataset
    test_zarr = ChunkedDataset(test_zarr_path).open()
    hand_dataset = HandwriteDataset(cfg, test_zarr, rasterizer, agents_mask_path=test_mask_path)
    test_dataset = StampDataset(hand_dataset)

    # wrap test dataset
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=args.gpus,
                                                                   rank=rank, shuffle=test_cfg["shuffle"])
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate_fn,
                                  num_workers=8, pin_memory=True, sampler=test_sampler)

    # Initialize network
    net = mfpNet(params)
    torch.cuda.set_device(gpu)
    net.cuda(gpu)

    # wrap model
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    # load pretrain model
    if test_cfg['model'] != "":
        save_cuda = test_cfg['cuda']
        net.load_state_dict(torch.load(test_cfg['model'], map_location={'cuda:%d' % save_cuda: 'cuda:%d' % gpu}))
        print('GPU{}: Finish importing pre-train model'.format(gpu))

    # calculate result
    net.eval()
    torch.set_grad_enabled(False)

    # store information for testing
    future_coords_offsets_pd = []
    timestamps = []
    agent_ids = []

    for data in test_data_loader:
        hist, nbrs, fut, mask, context, nbrs_info, index = data
        hist = hist.cuda(gpu)
        nbrs = nbrs.cuda(gpu)
        fut = fut.cuda(gpu)
        mask = mask.cuda(gpu)
        if context is not None:
            context = context.cuda(gpu)

        # Forward pass
        fut_preds, modes_pred = net.module.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, True, use_forcing=0)

        # transform predict to [modes, fut_len, num_agents, 2]
        fut_preds = torch.stack(fut_preds, 0)[:, :, :, :2]
