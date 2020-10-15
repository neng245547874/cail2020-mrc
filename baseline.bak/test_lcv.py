import json
import argparse
from tqdm import tqdm
from transformers import BertModel
from transformers import BertConfig as BC
import os
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
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
def predict_extra_sp(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=True):
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
            cur_sp_pred = []
            cur_id = batch['ids'][i]
            if cur_id >=100000:
                extra_index = feature_dict[cur_id].extra_sp_index- 5
                cur_id = cur_id-100000
                cur_sp_logit_pred = []  # for sp logit output
                for j in [5,predict_support_np.shape[1]]:
                    if (j+extra_index) >= len(example_dict[cur_id].sent_names):
                        break
                    if need_sp_logit_file:
                        temp_title, temp_id = example_dict[cur_id].sent_names[j]
                        cur_sp_logit_pred.append((temp_title, temp_id+extra_index, predict_support_np[i, j]))
                    if predict_support_np[i, j] > 0.5:

                        cur_sp_pred.append(example_dict[cur_id].sent_names[j+extra_index])
                        
                if cur_id not in sp_dict:
                    sp_dict.update({cur_id: cur_sp_pred})
                else:
                    sp_dict[cur_id].extend(cur_sp_pred)

            else:
                cur_sp_logit_pred = []  # for sp logit output
                for j in range(predict_support_np.shape[1]-1):
                    if j >= len(example_dict[cur_id].sent_names):
                        break
                    if need_sp_logit_file:
                        temp_title, temp_id = example_dict[cur_id].sent_names[j]
                        cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                    if predict_support_np[i, j] > args.sp_threshold:
                        cur_sp_pred.append(example_dict[cur_id].sent_names[j])
                if cur_id not in sp_dict:
                    sp_dict.update({cur_id: cur_sp_pred})
                else:
                    sp_dict[cur_id].extend(cur_sp_pred)


    new_answer_dict = {}
    for key, value in answer_dict.items():
        new_answer_dict[key] = value.replace(" ", "")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    temp_ids = []
    # for k, v in prediction["answer"].items():
    #     #     if v == "unknown":
    #     #         temp_ids.append(k)
    #     # for k, v in prediction["sp"].items():
    #     #     if k in temp_ids:
    #         prediction["sp"][k] = []
    with open(prediction_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)
    os.system(
        "python ../../evaluate/evaluate.py {} ./data/input/test.json".format(
            prediction_file))
    # os.system(
    #     "python ../evaluate/evaluate.py {} /data/wuyunzhao/CAIL2020/ydlj/baseline/data/input/test.json".format(
    #         prediction_file))

@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5

    for batch in tqdm(dataloader):
        query_mapping = batch['query_mapping']
        batch['context_mask'] = batch['context_mask'].float()
        loss, answer_loss, sp_loss, start_logits, end_logits, sp_logits = model(batch)
        outer = start_logits[:, :, None] + end_logits[:, None]  # None增加维度
        outer_mask = get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:  # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]  # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标   得到最高分
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        #unknown阈值
        # for i in range(outer.size()[0]):
            # null_score = outer[i,512,512] * 2
            # if outer.max(dim=2)[0].max(dim=1)[0][i] + outer.max(dim=1)[0].max(dim=1)[0][i] - null_score < 1:
            #     start_position[i] = 513
            #     end_position[i] = 513
        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                         start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), )
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            if example_dict[cur_id].question_text == "是否？？":
                cur_sp_pred.append(example_dict[cur_id].sent_names[2])
                cur_sp_pred.append(example_dict[cur_id].sent_names[5])
                cur_sp_pred.append(example_dict[cur_id].sent_names[6])
                cur_sp_pred.append(example_dict[cur_id].sent_names[7])
            # if example_dict[cur_id].sent_start_end_position[0][1] == 5:
            #     predict_support_np[i, 0] = 0
            else:
                for j in range(predict_support_np.shape[1]):
                    if j >= len(example_dict[cur_id].sent_names):
                        break


                    if need_sp_logit_file:
                        temp_title, temp_id = example_dict[cur_id].sent_names[j]
                        cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                    if predict_support_np[i, j] > args.sp_threshold:
                        cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    # new_answer_dict={}
    # for key,value in answer_dict.items():
    #     new_answer_dict[key]=value.replace(" ","")
    # with open("./output/predictions.json", 'r', encoding='utf-8') as reader:
    #     full_data = json.load(reader)

    prediction = {'answer':"", 'sp': sp_dict}
    #answer_confidence
    # temp_ids = []
    # for k, v in prediction["answer"].items():
    #     if v == "unknown":
    #         prediction["answer"][k] = "yes"
            # temp_ids.append(k)
    # for k, v in prediction["sp"].items():
    #     if k in temp_ids:
    #         prediction["sp"][k] = []
    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)
    os.system(
        "python ../../evaluate/evaluate.py {} ./data/input/test.json".format(prediction_file))

#coding: utf-8


if __name__ == '__main__':
    infile = "../input/data.json"
    outfile = "../result/result.json"

    os.system(
        "sh test_preprocess.sh {} ./data/output1".format(infile))
    # os.system("sh ./cail2019-1-master/test_base.sh")
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
    roberta_config = BC.from_pretrained(args.bert_model)
    encoder = BertModel.from_pretrained(args.bert_model, output_hidden_states=True)
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
            "{}".format("./output/submissions/test.json"))
