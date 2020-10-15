import argparse
import numpy as np
import os
import random
import sys
from os.path import join
from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from config import set_config
from model.modeling import *
from tools.data_helper import DataHelper
from tools.utils import convert_to_tokens

sys.path.append("../")
from evaluate.evaluate import eval

try:
    from apex import amp
except Exception:
    print('Apex not imoport!')
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import torch

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

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])

    sent_num_in_batch = batch["start_mapping"].sum()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1),
                                         batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3


import json


@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    test_loss_record = [0] * 3

    for batch in tqdm(dataloader):
        query_mapping = batch['query_mapping']
        batch['context_mask'] = batch['context_mask'].float()
        loss, answer_loss, sp_loss, start_logits, end_logits, sp_logits = model(batch)

        loss_list = [loss, answer_loss, sp_loss]
        if args.n_gpu > 1:
            loss_list[0] = loss_list[0].mean()
            loss_list[1] = loss_list[1].mean()
            loss_list[2] = loss_list[2].mean()

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                test_loss_record[i] += l.item()



        outer = start_logits[:, :, None] + end_logits[:, None]  # None增加维度
        outer_mask = get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if query_mapping is not None:  # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]  # 不允许预测query的内容

        # 这两句相当于找到了outer中最大值的i和j坐标   得到最高分
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                         start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), )
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

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
    metrics = eval(prediction_file, "/data/wuyunzhao/CAIL2020/ydlj/baseline/data/input/test_{}.json".format(k+1))
    # os.system(
    #     "python ../evaluate/evaluate.py {} /data/wuyunzhao/CAIL2020/ydlj/baseline/data/input/test.json".format(
    #         prediction_file))
    for i, l in enumerate(test_loss_record):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    return metrics

def train_epoch(data_loader, model, predict_during_train=False):
    global joint_f1
    # predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
    #         join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    model.train()
    pbar = tqdm(total=len(data_loader))
    epoch_len = len(data_loader)
    step_count = 0
    predict_step = epoch_len // 5

    while not data_loader.empty():
        step_count += 1
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float()
        train_batch(model, batch)
        del batch
        if predict_during_train and (step_count % predict_step == 0):
            current_joint_f1 = predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
                                       join(args.prediction_path,
                                            'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
            if current_joint_f1 > joint_f1:
                joint_f1 = current_joint_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(),
                           join(args.checkpoint_path, "best_model.pth"))
            model.train()
        pbar.update(1)

    metrics = predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
            join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    # if current_joint_f1 > joint_f1:
    #     joint_f1 = current_joint_f1
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     torch.save(model_to_save.state_dict(),
    #                join(args.checkpoint_path, "best_model.pth"))
    result[epc-1]['f1'] += metrics['f1']
    result[epc - 1]['sp_f1'] += metrics['sp_f1']
    result[epc - 1]['joint_f1'] += metrics['joint_f1']
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(),
               join(args.checkpoint_path, 'pred_seed_{}_epoch_{}_99999.pth'.format(args.seed, epc)))

def train_batch(model, batch):
    global global_step, total_train_loss, tr_loss

    loss, answer_loss, sp_loss, start_logits, end_logits, sp_logits = model(batch)
    # outer = start_logits[:, :, None] + end_logits[:, None]  # None增加维度
    # if query_mapping is not None:  # 这个是query_mapping (batch, 512)
    #     outer = outer - 1e30 * query_mapping[:, :, None]  # 不允许预测query的内容
    #
    # # 这两句相当于找到了outer中最大值的i和j坐标   得到最高分
    # start_position = outer.max(dim=2)[0].max(dim=1)[1]
    # end_position = outer.max(dim=1)[0].max(dim=1)[1]
    loss_list = [loss, answer_loss, sp_loss]
    if args.n_gpu > 1:
        loss_list[0] = loss_list[0].mean()
        loss_list[1] = loss_list[1].mean()
        loss_list[2] = loss_list[2].mean()

    if args.gradient_accumulation_steps > 1:
        loss_list[0] = loss_list[0] / args.gradient_accumulation_steps
        loss_list[1] = loss_list[1] / args.gradient_accumulation_steps
        loss_list[2] = loss_list[2] / args.gradient_accumulation_steps

    if args.fp16:
        with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))
        for i, l in enumerate(total_train_loss):
            print("Avg-LOSS{}/batch/step: {}".format(i, l / VERBOSE_STEP))
        total_train_loss = [0] * 3


if __name__ == "__main__":
    result = [{"f1":0, "sp_f1":0, "joint_f1":0} for i in range(10)]
    for k in range(5):
        parser = argparse.ArgumentParser()
        args = set_config(k+1)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.model_gpu
        args.n_gpu = torch.cuda.device_count()
        tr_loss = 0.0
        if args.seed == 0:
            args.seed = random.randint(0, 100)
            set_seed(args)

        helper = DataHelper(gz=True, config=args, k=str(k+1))
        args.n_type = helper.n_type  # 2

        # Set datasets
        Full_Loader = helper.train_loader
        # Subset_Loader = helper.train_sub_loader
        dev_example_dict = helper.dev_example_dict
        dev_feature_dict = helper.dev_feature_dict
        eval_dataset = helper.dev_loader
        joint_f1 = 0.0
        model = CailModel.from_pretrained(args.bert_model, output_hidden_states=True)
        #加载CAIL2019数据
        pretrained_dict = torch.load("/data/wuyunzhao/cail2019-1-master/save3/checkpoint-30968/pytorch_model.bin")
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load("/data/wuyunzhao/cail2019-1-master/save/checkpoint-3671/pytorch_model.bin"))
        # model.load_state_dict(torch.load("./output/checkpoints/train_v1/ckpt_seed_5_epoch_13_99999.pth"))
        if args.trained_weight is not None:
            model.load_state_dict(torch.load(args.trained_weight))
        model.to('cuda')

        # Initialize optimizer and criterions
        lr = args.lr
        t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
        warmup_steps = 0.1 * t_total

        param_optimizer = list(model.named_parameters())
        # for n in model.named_modules():
        #     print(n)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
        # binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        # sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        if args.fp16:
            import apex
            apex.amp.register_half_function(torch, "einsum")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.train()

        # Training
        global_step = epc = 0
        total_train_loss = [0] * 3

        VERBOSE_STEP = args.verbose_step
        while True:
            if epc == args.epochs:  # 5 + 30
                break
            epc += 1

            Loader = Full_Loader
            Loader.refresh()

            if epc > 2:
                train_epoch(Loader, model, predict_during_train=False)
            else:
                train_epoch(Loader, model)

    for i in range(10):
        result[i]["f1"] = result[i]["f1"] / 5
        result[i]["sp_f1"] = result[i]["sp_f1"] / 5
        result[i]["joint_f1"] = result[i]["joint_f1"] / 5
    for i, res in enumerate(result, start=1):
        print("epoch{}: avg_f1:{}, avg_sp_f1:{}, avg_joint_f1:{}".format(i, result[i - 1]["f1"],
                                                                         result[i - 1]["sp_f1"],
                                                                         result[i - 1]["joint_f1"]))
    with open("./result_k", 'a', encoding='utf8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)