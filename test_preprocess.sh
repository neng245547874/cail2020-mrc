source ~/.bashrc
INPUT_TRAIN_FILE=$1

OUTPUT_DIR=$2 #this dir must the same as the data_dir in train.sh

mkdir ${OUTPUT_DIR}
tokenizer_path='./chinese_roberta_wwm_large_ext_pytorch'

python data_process_extrasp.py \
    --tokenizer_path=$tokenizer_path \
    --full_data=${INPUT_TRAIN_FILE} \
    --example_output=${OUTPUT_DIR}/dev_example.pkl.gz \
    --feature_output=${OUTPUT_DIR}/dev_feature.pkl.gz \



