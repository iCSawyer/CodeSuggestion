with_exemplar=False
exemplar_len=-1
tgt_len=256
ATTN_TYPE=0
mem_len=$(($tgt_len - 1))
PRETRAIN="../datasets/save/pretrained/CodeGPT-small-java"
LANG="java"
DATASET="lucene_java_header_code_tokens"
model_dir="save/lucene_java_header_code_tokens/trxl_without_exemplar"
CUDA_VISIBLE_DEVICES=1 \
    python evaluate_line.py \
    --attn_type $ATTN_TYPE\
    --cuda \
    --model_dir $model_dir \
    --with_exemplar $with_exemplar \
    --data_dir ../datasets \
    --dataset $DATASET \
    --exemplar_len $exemplar_len \
    --tgt_len $tgt_len \
    --mem_len $mem_len \
    --eval_tgt_len $tgt_len \
    --pretrain $PRETRAIN \
    --lang $LANG \
    --eval_batch_size 1 \
    --n_exemplars 1