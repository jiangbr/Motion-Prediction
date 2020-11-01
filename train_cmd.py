# -*- coding: utf-8 -*-

"""
Main file of the whole project
"""

import argparse
import torch.multiprocessing as mp
from train import train


def main() -> None:
    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', '-g', type=int, default=1)
    parser.add_argument('--list', '-l', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--config', '-c', type=str, default='./config/config.gin')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train, test or validate')
    args = parser.parse_args()
    # call train or validate or test function
    if args.mode == 'train':
        mp.spawn(train, nprocs=args.gpus, args=(args, ))


if __name__ == "__main__":
    main()
