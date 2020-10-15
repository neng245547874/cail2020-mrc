export CUDA_VISIBLE_DEVICES=0

# for train and predict
python ./cail2019-1-master/run_cail_ensemble2.py \
--bert_model ./chinese_roberta_wwm_large_ext_pytorch \
--do_test \
--do_lower_case \
--test_file ./cail2019-1-master/output/test.json \
--max_seq_length 512 \
--doc_stride 128 \
--max_answer_length 128 \
--output_dir ./output \
--version_2_with_negative \
--predict_batch_size 16 \
--null_score_diff_threshold 100.0




