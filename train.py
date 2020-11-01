#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

import gin
import time
import numpy as np
from typing import *
from attrdict import AttrDict
from tqdm import tqdm

from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

from model_mfp import mfpNet
from my_utils import *
from MyDataset import MyDataset


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
        print("! gpu{}".format(rank) + logging_dir + " CREATED!")
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
    def __init__(self, my_dataset: MyDataset):
        """The dataset is used to transform items of MyDataset into desired format"""
        self.length = len(my_dataset)
        # define item list and assign transform
        self.dataset = my_dataset

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple:
        """Return corresponding item"""
        data = self.dataset[index]
        # define history trajectory
        hist = np.expand_dims(data['My_agent_history_position'], 1)
        # define future trajectory
        fut = np.expand_dims(data['target_positions'], 1)
        # define neighbor trajectory
        nbrs = np.transpose(data['neighbor_agents_history_position'], (1, 0, 2))
        # define mask
        mask = np.ones((fut.shape[0], fut.shape[1], 1)).astype(np.uint8)
        # define context
        context = None
        # define nbrs_info
        neighbor_list = [[x] for x in range(nbrs.shape[1])]
        nbrs_info = [{0: neighbor_list}]
        # combine to be a tuple
        item = (hist, nbrs, fut, mask, context, nbrs_info)

        return item

    @staticmethod
    def collate_fn(samples: List[Any]) -> Tuple:
        """Return desired format for DataLoader"""
        hist, nbrs, fut, mask, context, nbrs_info = samples[0]
        # transform to tensor
        hist = torch.from_numpy(hist).float()
        nbrs = torch.from_numpy(nbrs).float()
        fut = torch.from_numpy(fut).float()
        mask = torch.from_numpy(mask)
        if context is not None:
            context = torch.from_numpy(context)

        return hist, nbrs, fut, mask, context, nbrs_info


#####################################################################################################


def train(rank: int, args: Any) -> None:
    """Main training function"""

    # distributed training initialization
    gpu = args.list[rank]
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12222',
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
    my_dataset = MyDataset(cfg, train_zarr, rasterizer)

    # wrap dataset with StampDataset
    train_dataset = StampDataset(my_dataset)

    # wrap the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpus,
                                                                    rank=rank, shuffle=train_cfg["shuffle"])
    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=1, collate_fn=train_dataset.collate_fn,
                                   num_workers=0, pin_memory=True, sampler=train_sampler)

    # Initialize network
    net = mfpNet(params)
    torch.cuda.set_device(gpu)
    net.cuda(gpu)

    # wrap model
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    # define logging file
    logger_file, logging_dir = None, None
    if params.log:
        logger_file, logging_dir = setup_logger("./checkpts/", rank)

    train_loss: List = []

    # For efficiency, we first pre-train w/o interactive rollouts
    MODE = 'Pre'
    num_updates = 0
    optimizer = None

    for epoch_num in range(params.epoch):
        if MODE == 'EndPre':
            MODE = 'Train'
            print('Training with interactive rollouts.')
            bStepByStep = True
        else:
            print('Pre-training without interactive rollouts.')
            bStepByStep = False

        # Average losses.
        avg_tr_loss = []

        # begin training
        for data in train_data_loader:
            # transform from Pre mode to EndPre mode
            '''
            if num_updates > params.pre_train_num_updates and MODE == 'Pre':
                MODE = 'EndPre'
                break
            '''

            # Determine learning rate
            lr_fac = np.power(0.1, num_updates // params.updates_div_by_10)
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
            hist, nbrs, fut, mask, context, nbrs_info = data
            hist = hist.cuda(gpu)
            nbrs = nbrs.cuda(gpu)
            fut = fut.cuda(gpu)
            mask = mask.cuda(gpu)
            if context is not None:
                context = context.cuda(gpu)

            # Forward pass.
            fut_preds, modes_pred = net.module.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep)
            if params.modes == 1:
                l = nll_loss(fut_preds[0], fut, mask)
            else:
                l = nll_loss_multimodes(fut_preds, fut, mask, modes_pred)

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
                    msg_str_ = ("Epoch no:", epoch_num, "update:", num_updates, "| Avg train loss:",
                                format(np.mean(avg_tr_loss), '0.4f'), " learning_rate:%.5f" % lr)
                    msg_str = str([str(ss) for ss in msg_str_])
                    logger_file.write(msg_str + '\n')
                    logger_file.flush()

                avg_tr_loss = []
                '''
                if num_updates % params.iter_per_eval == params.iter_per_eval - 1:
                    print("Starting eval")
                    val_nll_err = evaluate('nll', net, params, val_data_loader, bStepByStep,
                                           use_forcing=params.use_forcing, y_mean=y_mean,
                                           num_batches=500, dataset_name='val_dl nll')

                    if params.log:
                        logger_file.write('val nll: ' + str(val_nll_err) + '\n')
                        logger_file.flush()
                '''

            # Save weights.
            if params.log and num_updates % params.iters_per_save == params.iters_per_save - 1 and rank == 0:
                msg_str = '\nSaving state, update iter:%d %s' % (num_updates, logging_dir)
                print(msg_str)
                logger_file.write(msg_str)
                logger_file.flush()
                torch.save(net.state_dict(),
                           logging_dir + '/checkpoints/ngsim_%06d' % num_updates + '.pth')  # type: ignore


def evaluate(metric: str, net: torch.nn.Module, params: AttrDict, data_loader: DataLoader, bStepByStep: bool,
             use_forcing: int, y_mean: torch.Tensor, num_batches: int, dataset_name: str) -> torch.Tensor:
    """Evaluation function for validation and test data.

    Given a MFP network, data loader, evaluate either NLL or RMSE error.
    """
    print('eval ', dataset_name)
    num = params.fut_len_orig_hz // params.subsampling
    lossVals = torch.zeros(num)
    counts = torch.zeros(num)

    for i, data in enumerate(data_loader):
        if i >= num_batches:
            break
        hist, nbrs, mask, fut, mask, context, nbrs_info = data
        if params.use_cuda:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            fut = fut.cuda()
            mask = mask.cuda()
            if context is not None:
                context = context.cuda()

        l, c = None, None
        if metric == 'nll':
            fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep,
                                                    use_forcing=use_forcing)
            if params.modes == 1:
                if params.remove_y_mean:
                    fut_preds[0][:, :, :2] += y_mean.unsqueeze(1).to(fut.device)
                l, c = nll_loss_test(fut_preds[0], fut, mask)
            else:
                l, c = nll_loss_test_multimodes(fut_preds, fut, mask, modes_pred, y_mean.to(fut.device))
        else:  # RMSE error
            assert params.modes == 1
            fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep,
                                                    use_forcing=use_forcing)
            if params.modes == 1:
                if params.remove_y_mean:
                    fut_preds[0][:, :, :2] += y_mean.unsqueeze(1).to(fut.device)
                l, c = mse_loss_test(fut_preds[0], fut, mask)

        lossVals += l.detach().cpu()
        counts += c.detach().cpu()

    if metric == 'nll':
        err = lossVals / counts
        print(lossVals / counts)
    else:
        err = torch.pow(lossVals / counts, 0.5) * 0.3048
        print(err)  # Calculate RMSE and convert from feet to meters
    return err
