date
bert_dir='/data/wuyunzhao/pre-model/chinese_roberta_wwm_large_ext_pytorch'
python run_cail.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output \
    --batch_size 2 \
    --eval_batch_size 2 \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --epochs 10 \
    --model_gpu 2,3\
    --seed 5 \
    --fp16
date
