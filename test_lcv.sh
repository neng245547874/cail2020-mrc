date
bert_dir='./chinese_roberta_wwm_large_ext_pytorch'
python test_ensemble2.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir ./data/output \
    --batch_size 2 \
    --eval_batch_size 12  \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --epochs 10 \
    --trained_weight "./output/checkpoints/train_v1/pred_seed_5_epoch_8_99999.pth"\
    --sp_threshold 0.5 \
    --model_gpu 0

date
