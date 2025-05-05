#!/bin/bash

# 定义日志文件
log_file="train.txt"


# 如果需要将输出同时显示在屏幕上和记录到文件中，可以使用 tee
python -m src.main +experiment=acid mode=test wandb.name=acid dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=./pretrained_weights/mamba.ckpt test.save_image=true 
