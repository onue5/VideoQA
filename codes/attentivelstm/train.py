"""
Train CNN-based model

Usage:
    python train.py --datadir <data dir> --gpuids <gpu_ids>

Examples:
    python train.py --datadir ../../data --gpuids 0
"""

import sys
sys.path.append('../../../VideoQAModelDS/')
import os
import argparse
from codes.attentivelstm.main import train


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--datadir', help='source data directory')
    # parser.add_argument('--gpuids', help='cuda visible devices')
    #
    # parser.set_defaults(gpuids="0")
    # args = parser.parse_args()
    #
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids
    #
    # data_dir = args.datadir

    data_dir = "../../data"

    train(
        video_fpath=os.path.join(data_dir, 'videos.json'),
        train_fpath=os.path.join(data_dir, 'train.json'),
        dev_fpath=os.path.join(data_dir, 'dev.json'),
        vocab_fpath=os.path.join(data_dir, 'vocab.txt'),
        # num_iter=30000,
        embedding_fpath=os.path.join(data_dir, 'vocab_embedding.txt'),
        num_epoch=20
        # train_period=2000,
        # dev_period=2000
    )

