date
bert_dir='./chinese_roberta_wwm_large_ext_pytorch'
python ./baseline.bak/test.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output1 \
    --batch_size 2 \
    --eval_batch_size 32 \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --trained_weight "./baseline.bak/output/checkpoints/train_v1/ckpt_seed_5_epoch_10_99999.pth" \
    --epochs 10 \
#    --model_gpu 2,3

date
