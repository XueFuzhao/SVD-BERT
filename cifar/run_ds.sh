#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export PATH=~/.local/bin:$PATH
deepspeed cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@
