lang='py' 
dataset="django"
pretrain_dir="../save/django/with_exemplar"
with_exemplar=True
tgt_len=512
exemplar_len=64
gpus=0
gpu_num=$(echo "$gpus" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$gpus python evaluate.py \
        --dataset $dataset \
        --lang $lang \
        --pretrain_dir $pretrain_dir \
        --model_type=gpt2 \
        --tgt_len $tgt_len \
        --exemplar_len $exemplar_len \
        --do_eval \
        --gpu_per_node $gpu_num \
        --learning_rate 1e-5 \
        --weight_decay=0.01 \
        --max_grad_norm 1 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=50 \
        --logging_steps=100 \
        --save_steps=1000 \
        --overwrite_output_dir \
        --per_gpu_eval_batch_size=1 \
        --with_exemplar $with_exemplar \
        --n_exemplars 1