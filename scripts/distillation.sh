export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
python -m torch.distributed.run --nnodes ${SLURM_NNODES} \
        --node_rank \$SLURM_PROCID \
        --nproc_per_node=${SLURM_GPUS_PER_NODE} \
        --rdzv-id=${SLURM_JOBID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py \
--training_type distillation \
--echo \
--multilingual_encoder_checkpoint BAAI/bge-m3-unsupervised \
--lsr_encoder_checkpoint naver/splade-v3 \
--pretrained_alignment_checkpoint ./checkpoints/milco-alignment-mmarco-32 \
--train_datasets bge-distillation \
--train_group_size 8 \
--query_max_length 64 \
--passage_max_length 256 \
--lambda_q 1e-3 \
--lambda_d 1e-5 \
--output_dir ./checkpoints/milco-distillation-bge \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 128 \
--num_train_epochs 8 \
--bf16 \
--save_total_limit  5 \
--warmup_ratio 0.03 \
--lr_scheduler_type 'cosine' \
--report_to 'wandb' \
--dataloader_num_workers 1 \
--learning_rate 2e-5 \
--logging_steps 500 \
--eval_strategy 'steps' \
--save_steps 20 \
--eval_steps 10   \
--eval_languages hi zh yo \
--eval_datasets miracl_hard_negatives \
--eval_top_k_candidates 100