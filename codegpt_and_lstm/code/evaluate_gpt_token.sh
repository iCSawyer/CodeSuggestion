gpus=3
gpu_num=$(echo "$gpus" | tr ',' '\n' | wc -l)
lang='java' 
with_exemplar=False
tgt_len=256
exemplar_len=-1
dataset="lucene_java_header_code_tokens"
pretrain_dir="../save/lucene_java_header_code_tokens/without_exemplar/lstm_new/checkpoint-best"
CUDA_VISIBLE_DEVICES=$gpus python eval.py \
        --with_exemplar $with_exemplar \
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
        --per_gpu_eval_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --num_train_epochs=50 \
        --logging_steps=100 \
        --save_steps=1000 \
        --overwrite_output_dir
python token_evaluator.py --lang $lang --dataset $dataset --remove "no"