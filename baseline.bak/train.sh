export CUDA_VISIBLE_DEVICES="3"
date
bert_dir='/home/wuyunzhao/CAIL2020/ydlj/baseline/chinese_roberta_wwm_large_ext_pytorch'
python run_cail.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output \
    --batch_size 2 \
    --eval_batch_size 8 \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --sp_lambda 1\
    --epochs 10 \
    --seed 5 \
    --fp16
date
