"""
Train a diffusion model on images.
"""
import sys
# put your path here
sys.path.extend(['/home/csantiago/Dif-fuse'])

from utils.gpu_selection_utils import set_gpu_visible_devices
set_gpu_visible_devices(2)

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.VinDrMammo_dataset import *
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import os
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader


def load_data(loader):
    while True:
        yield from loader


def main():
    args = create_argparser().parse_args()
    print(vars(args))

    dist_util.setup_dist()
    logger.configure(experiment_name=args.experiment_name)

    args_path = os.path.join(logger.get_dir(), 'args.txt')
    with open(args_path, 'w') as convert_file:
        convert_file.write(json.dumps(vars(args)))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    device = dist_util.dev()
    print("Using device:", device)
    model = model.to(device)
    if args.gpus > 1:
        model = nn.DataParallel(model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    train_set = VinDrMammoDataset(
        dataset_root_folder_filepath='/home/csantiago/data',
        df_path='data/grouped_df_train.csv',
        transform=None,
        only_positive=False,
        only_negative=True)

    loader = DataLoader(
         train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    data = load_data(loader)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps
    ).run_loop()


def create_argparser():
    defaults = dict(
        gpus=2,
        experiment_name='test',
        schedule_sampler="uniform", # Futuramente experimentar com linear e cossine
        lr=1e-4,
        weight_decay=0.05,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=4,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":

    main()
