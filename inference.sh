export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export CHIEF_IP=127.0.0.2
export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

path=./
train_path=$path/run_clm_llms_inference.py
    
torchrun --nnodes 1 --nproc_per_node 1 \
    ${train_path} \
    --deepspeed $path/configs/deepspeed_config.json \
    --train_file $path/data/train_total_new_name.cache \
    --model_name_or_path ${path} \
    --dataset_name vqa \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --do_eval \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing False \
    --output_dir $path/trained_models/MM-LLMs/mm_llms_trainer/
