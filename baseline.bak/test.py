import json
import argparse
from os.path import join
from tqdm import tqdm
from transformers import BertModel
from transformers import BertConfig as BC
import os
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper

try:
    from apex import amp
except Exception:
    print('Apex not import!')

cache_S = 0
cache_mask = None


def get_output_mask(outer):  # 挡住一些位置
    global cache_S, cache_mask
    S = outer.size(1)
    if S <= cache_S:
        return Variable(cache_mask[:S, :S], requires_grad=False)
    cache_S = S
    # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
    np_mask = np.tril(np.triu(np.ones((S, S)), 0), 128)
    cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
    return Variable(cache_mask, requires_grad=False)


@torch.no_grad()
def predict_extra_sp(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    test_loss_record = [0] * 3

    for batch in tqdm(dataloader):
        query_mapping = batch['query_mapping']
        batch['context_mask'] = batch['context_mask'].float()
        sp_logits = model(batch)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []  # for sp logit output
            cur_id = batch['ids'][i]
            if example_dict[cur_id].question_text == "是否？？":
                cur_sp_pred.append(example_dict[cur_id].sent_names[2])
                cur_sp_pred.append(example_dict[cur_id].sent_names[5])
                cur_sp_pred.append(example_dict[cur_id].sent_names[6])
                cur_sp_pred.append(example_dict[cur_id].sent_names[7])
            else:
                for j in range(predict_support_np.shape[1]):
                    if j >= len(example_dict[cur_id].sent_names):
                        break
                    if predict_support_np[i, j] > 0.1:
                        cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            if cur_id not in sp_dict:
                sp_dict.update({cur_id: cur_sp_pred})
            else:
                sp_dict[cur_id].extend(cur_sp_pred)
    with open("./output/predictions.json", 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    prediction = {'answer': full_data["answer"], 'sp': sp_dict}


    # for k, v in prediction["answer"].items():
    #     #     if v == "unknown":
    #     #         temp_ids.append(k)
    #     # for k, v in prediction["sp"].items():
    #     #     if k in temp_ids:
    #         prediction["sp"][k] = []
    with open("./output/sp_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)

# coding: utf-8


if __name__ == '__main__':
    infile = "../input/data.json"
    outfile = "../result/result.json"
    os.system(
        "sh ./baseline.bak/test_preprocess.sh {} ./data/output1".format(infile))

    parser = argparse.ArgumentParser()
    args = set_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.model_gpu
    args.n_gpu = torch.cuda.device_count()

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    # Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader
    roberta_config = BC.from_pretrained(args.bert_model, output_hidden_states=True)
    encoder = BertModel(roberta_config)

    args.input_dim = roberta_config.hidden_size
    model = BertSupportNet(config=args, encoder=encoder)
    # model = CailModel.from_pretrained(args.bert_model, output_hidden_states=True)
    if args.trained_weight is not None:
        model.load_state_dict(torch.load(args.trained_weight))
    model.to('cuda')
    # Initialize optimizer and criterions
    # lr = args.lr
    # t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    # warmup_steps = 0.1 * t_total
    # optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                             num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    # if args.fp16:
    #     import apex
    #
    #     apex.amp.register_half_function(torch, "einsum")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # model = torch.nn.DataParallel(model)

    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    VERBOSE_STEP = args.verbose_step

    predict_extra_sp(model, eval_dataset, dev_example_dict, dev_feature_dict,
                     outfile)
