export MASTER_ADDR=$(hostname)
export MASTER_PORT=25900
python -m torch.distributed.run --nnodes ${SLURM_NNODES} \
        --node_rank \$SLURM_PROCID \
        --nproc_per_node=${SLURM_GPUS_PER_NODE} \
        --rdzv-id=${SLURM_JOBID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
train.py \
--training_type alignment \
--multilingual_encoder_checkpoint BAAI/bge-m3-unsupervised \
--lsr_encoder_checkpoint naver/splade-v3 \
--train_datasets mmarco_passage mmarco_query wikimatrix europarl opensubtitles talks tatoeba jw300 news-commentary \
--max_length 512 \
--output_dir ./checkpoints/milco-alignment-mmarco-32 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128 \
--num_train_epochs 1 \
--learning_rate 2e-5 \
--bf16 \
--logging_steps 500 \
--eval_strategy steps \
--eval_steps 5000 \
--save_steps 5000 \
--eval_languages hi zh yo \
--eval_top_k_candidates 100 \
--report_to wandb \
--dynamic_length
