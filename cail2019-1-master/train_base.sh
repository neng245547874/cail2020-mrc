export CUDA_VISIBLE_DEVICES=1

# for train and predict
# bert-base-chinese
# /home/delaiq/data/bert/wwm
python run_cail.py \
--bert_model /data/wuyunzhao/pre-model/chinese_roberta_wwm_large_ext_pytorch \
--do_train \
--do_lower_case \
--version_2_with_negative \
--train_file /data/wuyunzhao/mydata/data/cail2020train.json \
--dev_file /data/wuyunzhao/mydata/data/cail2020test.json \
--train_batch_size 1 \
--predict_batch_size 1 \
--learning_rate 2e-5 \
--num_train_epochs 5.0 \
--max_seq_length 512 \
--doc_stride 128 \
--max_answer_length 128 \
--output_dir /data/wuyunzhao/cail2019-1-master/save2020 \
#--fp16