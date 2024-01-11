LANG="py"
DATASET="lucene_py_comment_comment"
PRETRAIN="datasets/save/pretrained/CodeGPT-small-py"
with_exemplar=True
exemplar_len=255
tgt_len=512
mem_len=$((tgt_len - 1))
CUDA_VISIBLE_DEVICES=1 \
    python train.py \
    --debug\
    --with_exemplar $with_exemplar \
    --cuda \
    --data_dir datasets \
    --dataset $DATASET \
    --n_layer 18 \
    --d_model 1024 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 3072 \
    --dropout 0.0 \
    --dropatt 0.0 \
    --lr 75e-6 \
    --warmup_step 0 \
    --max_step 1000000 \
    --exemplar_len $exemplar_len \
    --tgt_len $tgt_len \
    --mem_len $mem_len \
    --eval_tgt_len $tgt_len \
    --batch_size 8 \
    --eval_batch_size 8 \
    --log-interval 200 \
    --eval-interval 2000 \
    --pretrain $PRETRAIN \
    --lang $LANG