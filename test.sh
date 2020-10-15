date
bert_dir='./chinese_roberta_wwm_large_ext_pytorch'
python test_ensemble.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output1 \
    --batch_size 2 \
    --eval_batch_size 8 \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --epochs 10 \
    --trained_weight "./output/checkpoints/train_v1/pred_seed_10_epoch_8_99999.pth" \
    --sp_threshold 0.1 \
    --model_gpu 0 \
#    --model_gpu 2,3

date
