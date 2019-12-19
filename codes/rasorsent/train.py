"""
Train our model

Usage:
    python train.py --model <model name> --datadir <data_dir> --gpuids <gpu_ids>

Examples:
    python train.py --model ourmodel --datadir ../../data_test --gpuids 0
"""

import sys
sys.path.append('../../')
import os
import argparse
from codes.rasorsent.main import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name')
    parser.add_argument('--datadir', help='source data directory')
    parser.add_argument('--gpuids', help='cuda visible devices')

    parser.set_defaults(gpuids="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids

    model = args.model
    data_dir = args.datadir

    output_dir = "../run/{}".format(model)

    if model == "ourmodel":
        train(
            model_name="ourmodel",
            video_fpath=os.path.join(data_dir, 'videos.json'),
            train_fpath=os.path.join(data_dir, 'train.json'),
            dev_fpath=os.path.join(data_dir, 'dev.json'),
            vocab_fpath=os.path.join(data_dir, 'vocab.txt'),
            output_dir=os.path.join(output_dir, 'models'),
            embedding_fpath=os.path.join(data_dir, 'vocab_embedding.txt'),
            num_epoch=100,
        )


# if __name__ == "__main__":
#     data_dir = "../../data_test/"
#     output_dir = "../run/{}".format("ourmodel")
#
#     train(
#         model_name="iwan",
#         video_fpath=os.path.join(data_dir, 'videos.json'),
#         train_fpath=os.path.join(data_dir, 'train.json'),
#         dev_fpath=os.path.join(data_dir, 'dev.json'),
#         vocab_fpath=os.path.join(data_dir, 'vocab.txt'),
#         output_dir=os.path.join(output_dir, 'models'),
#         embedding_fpath=os.path.join(data_dir, 'vocab_embedding.txt'),
#         # num_epoch=20,
#         num_epoch=20,
#     )






