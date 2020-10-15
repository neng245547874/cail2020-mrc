import argparse
import numpy as np
import os
import random
from os.path import join
from tqdm import tqdm
from transformers import BertConfig as BC
from transformers import BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from config import set_config
from tools.data_helper import DataHelper
from tools.data_iterator_pack import IGNORE_INDEX
from tools.utils import convert_to_tokens

try:
    from apex import amp
except Exception:
    print('Apex not imoport!')


import torch
from torch import nn


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


def compute_loss(batch, sp_logits):

    sent_num_in_batch = batch["start_mapping"].sum()
    # print(sent_num_in_batch)
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.contiguous().view(-1), batch['is_support'].contiguous().float().view(-1)).sum() / sent_num_in_batch

    return loss3



import json
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

        # loss_list = [loss, answer_loss, sp_loss]


        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            # print(predict_support_np.shape[0])
            cur_sp_pred = []
            cur_id = batch['ids'][i]
            if cur_id >=100000:
                extra_index = feature_dict[cur_id].extra_sp_index-5
                cur_id = cur_id-100000
                cur_sp_logit_pred = []  # for sp logit output
                for j in [5,predict_support_np.shape[1]]:
                    if (j+extra_index) >= len(example_dict[cur_id].sent_names):
                        break
                    if need_sp_logit_file:
                        temp_title, temp_id = example_dict[cur_id].sent_names[j]
                        cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                    if predict_support_np[i, j] > args.sp_threshold:
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
        batch['context_mask'] = batch['context_mask'].float()
        sp_logits = model(batch)


        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], [0 for i in range(len(batch))],
                                         [0 for i in range(len(batch))], np.ones(len(batch)))
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

    new_answer_dict={}
    for key,value in answer_dict.items():
        new_answer_dict[key]=value.replace(" ","")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)
    os.system(
        "python ../../evaluate/evaluate.py {} ./data/input/test.json".format(prediction_file))


def train_epoch(data_loader, model, predict_during_train=False):
    # predict_extra_sp(model, eval_dataset, dev_example_dict, dev_feature_dict,
    #         join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.js,on'.format(args.seed, epc)))
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
            predict_extra_sp(model, eval_dataset, dev_example_dict, dev_feature_dict,
                     join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
            # model_to_save = model.module if hasattr(model, 'module') else model
            # torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pth".format(args.seed, epc, step_count)))
            model.train()
        pbar.update(1)

    predict_extra_sp(model, eval_dataset, dev_example_dict, dev_feature_dict,
             join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
   


def train_batch(model, batch):
    global global_step, total_train_loss

    sp_logits = model(batch)
    # print(sp_logits)
    # if batch["ids"][0]>100000:
    #     # print(batch["context_idxs"][0],batch["context_mask"][0],batch["segment_idxs"][0])
    #     print(batch["start_position"])
    loss = compute_loss(batch, sp_logits)

    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    global_step += 1


    total_train_loss += loss.item()

    if global_step % VERBOSE_STEP == 0:
        print("{} -- In Epoch{}: ".format(args.name, epc))

        print("Avg-LOSS/batch/step: {}".format(total_train_loss / VERBOSE_STEP))

        total_train_loss = 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    args.n_gpu = torch.cuda.device_count()

    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    Full_Loader = helper.train_loader
    # Subset_Loader = helper.train_sub_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader



    roberta_config = BC.from_pretrained(args.bert_model)
    encoder = BertModel.from_pretrained(args.bert_model, output_hidden_states=True)
    # pretrained_dict = torch.load("/data/wuyunzhao/cail2019-1-master/save3/checkpoint-30968/pytorch_model.bin")
    # encoder.load_state_dict(pretrained_dict, strict=False)
    # model_dict = encoder.state_dict()
    # model_dict.update(pretrained_dict)
    # encoder.load_state_dict(model_dict)
    args.input_dim=roberta_config.hidden_size
    model = BertSupportNet(config=args, encoder=encoder)

    model.to('cuda')

    # Initialize optimizer and criterions
    lr = args.lr
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = 0.1 * t_total
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  

    if args.fp16:
        import apex
        apex.amp.register_half_function(torch, "einsum")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    if args.n_gpu > 1:

        model = torch.nn.DataParallel(model)
    model.train()

    # Training
    global_step = epc = 0
    total_train_loss = 0
    test_loss_record = []
    VERBOSE_STEP = args.verbose_step
    while True:
        if epc == args.epochs:  # 5 + 30
            exit(0)
        epc += 1

        Loader = Full_Loader
        Loader.refresh()

        if epc > 2:
            train_epoch(Loader, model, predict_during_train=False)
        else:
            train_epoch(Loader, model)
            
    model_to_save = model.half().module if hasattr(model, 'module') else model.half()
    torch.save(model_to_save.state_dict(),
               join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pth".format(args.seed, epc)))

