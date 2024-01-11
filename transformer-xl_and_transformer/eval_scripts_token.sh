PRETRAIN="../datasets/save/pretrained/CodeGPT-small-java"
LANG="java"
DATASET="lucene_java_header_code_tokens"
model_dir="save/lucene_java_comment_code_tokens/tfm_without_exemplar"
with_exemplar=False
exemplar_len=-1
tgt_len=256
mem_len=0
ATTN_TYPE=2
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_token.py \
    --attn_type $ATTN_TYPE \
    --cuda \
    --model_dir $model_dir \
    --with_exemplar $with_exemplar \
    --data_dir ../datasets \
    --dataset $DATASET \
    --exemplar_len $exemplar_len \
    --tgt_len $tgt_len \
    --mem_len $mem_len \
    --eval_tgt_len $tgt_len \
    --eval_batch_size 1 \
    --pretrain $PRETRAIN \
    --lang $LANG
python evaluate_token_helper.py --lang $LANG --dataset $DATASET  --remove "no"