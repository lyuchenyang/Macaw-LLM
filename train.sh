export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

# export MASTER_ADDR="${CHIEF_IP:=localhost}"
# export CHIEF_IP=127.0.0.1
# export MASTER_ADDR="${CHIEF_IP:=localhost}"
# export MASTER_PORT="${MASTER_PORT:=29500}"

# yum install ffmpeg -y
# pip install peft

path=/apdcephfs/share_733425/vinnylywang/georgelv
train_path=$path/run_clm_llms.py

torchrun --nnodes 1 --nproc_per_node 8 \
    ${train_path} \
    --deepspeed $path/train/deepspeed_config.json \
    --model_name_or_path ${path} \
    --llm_model_name_or_path trained_models/vicuna_model/ \
    --tokenizer_name trained_models/vicuna_model/ \
    --train_file $path/data/train_total_new_instruction_vicuna.cache \
    --image_instruction_file data/generated_examples_coco_1.json \
    --video_instruction_file data/generated_examples_avsd_1.json \
    --visual_names_file data/all_visual_names_vicuna_new.json \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 3 \
    --num_train_epochs 5 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --block_size 512 \
    --n_frames 6 \
    --attention_heads 8 \
    --image_conv_kernel 36 \
    --image_conv_stride 6 \
    --video_conv_kernel 48 \
    --video_conv_stride 12 \
    --audio_conv_kernel 180 \
    --audio_conv_stride 120 \
    --freeze_multi_modal_encoder False \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing False \
    --output_dir $path/trained_models/MM-LLMs/mm_llms_trainer_1/
