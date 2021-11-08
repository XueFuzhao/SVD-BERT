#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PATH=~/.local/bin:$PATH



# Number of nodes
NUM_NODES=1
# Number of GPUs per node
NUM_GPUS=1
# Size of expert parallel world (should be less than total world size)
EP_SIZE=1
# Number of total experts
EXPERTS=4

deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} cifar10_deepspeed.py \
	--log-interval 1000 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size ${EP_SIZE} \
	--num-experts ${EXPERTS} \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group \
        --model_path='/home/users/nus/e0792473/scratch/DeepSpeedExamples/cifar/model/moe_cnn'
