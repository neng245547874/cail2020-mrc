date
bert_dir='../chinese_roberta_wwm_large_ext_pytorch'
python test_lcv.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output1 \
    --batch_size 2 \
    --eval_batch_size 12  \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --epochs 10 \
    --trained_weight "./baseline.bak/output/checkpoints/train_v1/ckpt_seed_5_epoch_2_99999.pth"\
    --sp_threshold 0.5 \
    --model_gpu 0

date
