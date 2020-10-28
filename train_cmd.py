# -*- coding: utf-8 -*-

"""
Main file of the whole project
"""

import gin
import argparse
from train import train, Params


def main() -> None:
    # define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='./config/config.gin')
    args = parser.parse_args()
    # import config with gin
    gin.parse_config_file(args.config)
    params = Params()()
    train(params)


if __name__ == "__main__":
    main()
