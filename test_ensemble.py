import json
import argparse
from tqdm import tqdm
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
from data_process import InputFeatures,Example
from collections import Counter
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
def predict(example_dict, prediction_file):

    answer_dict = {}
    with open("./output/sp_predictions.json", 'r', encoding='utf-8') as reader:
        sp_data = json.load(reader)
    sp_dict = {}
    sp_dict1 = sp_data["sp"]
    total_test_loss = [0] * 5

    for id, sp_list in count_dict.items():
        cur_sp_pred = []
        if example_dict[id].question_text == "是否？？":
            cur_sp_pred.append(example_dict[id].sent_names[2])
            cur_sp_pred.append(example_dict[id].sent_names[5])
            cur_sp_pred.append(example_dict[id].sent_names[6])
            cur_sp_pred.append(example_dict[id].sent_names[7])
        else:
            sp_list = Counter(sp_list)
            for sp, count in sp_list.items():
                if len(list(zip(*sp_dict1[str(id)]))) > 0:
                    if sp in list(zip(*sp_dict1[str(id)]))[1]:
                        count +=3
                if count >= 7:
                    cur_sp_pred.append(example_dict[id].sent_names[sp])

        # sp_dict[str(id)].extend(cur_sp_pred)
        sp_dict.update({id: cur_sp_pred})

    with open("./output/predictions.json", 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    prediction = {'answer': full_data["answer"], 'sp': sp_dict}

    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)
    # os.system(
    #     "python ../evaluate/evaluate.py {} /data/wuyunzhao/CAIL2020/ydlj/baseline/data/input/test.json".format(prediction_file))

@torch.no_grad()
def sub_predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model.eval()


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
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                         start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), )
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            if cur_id not in count_dict:
                count_dict[cur_id] = []
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > args.sp_threshold: #投票
                    count_dict[cur_id].append(j)




if __name__ == '__main__':
    infile = "../input/data.json"
    outfile = "../result/result.json"
    os.system("sh ./cail2019-1-master/test_base.sh")
    os.system("sh ./baseline.bak/test.sh")
    os.system(
        "sh test_preprocess.sh {} ./data/output1".format(infile))

    bert_models = ['./output/checkpoints/train_S1/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S2/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S3/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S4/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S5/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S6/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S7/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S8/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S9/pred_seed_5_epoch_10_99999.pth',
                   './output/checkpoints/train_S10/pred_seed_5_epoch_10_99999.pth', ]
    answer_dict = {}
    count_dict = {}
    for model_dir in bert_models:
        parser = argparse.ArgumentParser()
        args = set_config(k='')
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
        config = BC.from_pretrained('./chinese_roberta_wwm_large_ext_pytorch', output_hidden_states=True)
        model = CailModel(config)
        pretrained_dict = torch.load(model_dir)
        model.load_state_dict(pretrained_dict, strict=False)
        # model = CailModel.from_pretrained(args.bert_model, output_hidden_states=True)
        model.to('cuda')
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

        sub_predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                outfile)

    predict(dev_example_dict, outfile)