LOG_DIR="log"
if [ ! -d "$LOG_DIR" ]; then
  mkdir $LOG_DIR
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

NGPU=1
#export CUDA_VISIBLE_DEVICES=3


echo "Started scripts"

TASK=MNLI
EFFECTIVE_BATCH_SIZE=32
LR=2e-5
NUM_EPOCH=3
MASTER_PORT=1231

model_name="bert_base"
JOBNAME=$TASK
CHECKPOINT_PATH="/home/users/nus/e0792473/scratch/output_svd_bert/bert_model_outputs/saved_models/baseline_pretraining/epoch32_step400/mp_rank_00_model_states.pt"
OUTPUT_DIR="/home/users/nus/e0792473/scratch/outputs/${model_name}/${JOBNAME}_bsz${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}"

GLUE_DIR="/home/users/nus/e0792473/scratch/GLUE-baselines/glue_data"

MAX_GPU_BATCH_SIZE=32
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi

echo "Fine Tuning $CHECKPOINT_PATH"
run_cmd="python -m torch.distributed.launch \
       --nproc_per_node=${NGPU} \
       --master_port=${MASTER_PORT} \
       run_glue_classifier_bert_base.py \
       --task_name $TASK \
       --do_train \
       --do_eval \
       --deepspeed \
       --deepspeed_config glue_bert_base.json \
       --do_lower_case \
       --fp16 \
       --data_dir $GLUE_DIR/$TASK/ \
       --bert_model bert-base-uncased \
       --max_seq_length 128 \
       --train_batch_size ${PER_GPU_BATCH_SIZE} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --learning_rate ${LR} \
       --num_train_epochs ${NUM_EPOCH} \
       --output_dir ${OUTPUT_DIR}_${TASK} \
       --model_file $CHECKPOINT_PATH &> $LOG_DIR/${model_name}/${JOBNAME}_${TASK}_bzs${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}.txt
       "
echo ${run_cmd}
eval ${run_cmd} 


#       --deepspeed_transformer_kernel \
#       --progressive_layer_drop \
