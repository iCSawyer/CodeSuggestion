lang='py' 
dataset="django"
pretrain_dir="../save/CodeGPT-small-py-adaptedGPT2"
with_exemplar=True
tgt_len=512
exemplar_len=200
gpus=0
gpu_num=$(echo "$gpus" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node $gpu_num --master_port 5235 train.py \
        --with_exemplar $with_exemplar \
        --dataset $dataset \
        --lang $lang \
        --pretrain_dir $pretrain_dir \
        --model_type=gpt2 \
        --tgt_len $tgt_len \
        --exemplar_len $exemplar_len \
        --do_train \
        --gpu_per_node $gpu_num \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --max_grad_norm 1 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=8 \
        --per_gpu_eval_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --num_train_epochs=100 \
        --logging_steps=500 \
        --save_steps=500 \
        --overwrite_output_dir
