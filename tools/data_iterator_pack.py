import numpy as np
import torch
from numpy.random import shuffle

IGNORE_INDEX = -100


class DataIteratorPack(object):
    def __init__(self, features, example_dict,bsz, device, sent_limit, entity_limit,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz   # batch_size
        self.device = device
        self.features = features
        self.example_dict = example_dict
        # self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        # self.para_limit = 4 
        # self.entity_limit = entity_limit
        self.example_ptr = 0
        self.max_length = 512
        if not sequential:
            shuffle(self.features)  

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, self.max_length).to("cuda")
        context_mask = torch.LongTensor(self.bsz, self.max_length).to("cuda")
        segment_idxs = torch.LongTensor(self.bsz, self.max_length).to("cuda")
       

        query_mapping = torch.Tensor(self.bsz, self.max_length+3).to("cuda") #预测答案时候mask掉query
        query_mapping2 = torch.Tensor(self.bsz, self.max_length).to("cuda")
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, self.max_length).to("cuda") #确定句子个数 做损失
        end_mapping = torch.Tensor(self.bsz, self.sent_limit, self.max_length).to("cuda")
        all_mapping = torch.Tensor(self.bsz, self.max_length, self.sent_limit).to("cuda") #预测证据时用到

        start_position = [[0 for i in range(self.sent_limit)] for i in range(self.bsz)]
        end_position = [[0 for i in range(self.sent_limit)] for i in range(self.bsz)]
        # Label tensor
        y1 = torch.LongTensor(self.bsz).to("cuda")
        y2 = torch.LongTensor(self.bsz).to("cuda")
        q_type = torch.LongTensor(self.bsz).to("cuda")
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).to("cuda")


        

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr  
            cur_bsz = min(self.bsz, len(self.features) - start_id)   #读取batch_size
            cur_batch = self.features[start_id: start_id + cur_bsz]  
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)  

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0

            start_position = [[0 for i in range(self.sent_limit)] for i in range(self.bsz)]
            end_position = [[0 for i in range(self.sent_limit)] for i in range(self.bsz)]

            for mapping in [start_mapping, all_mapping,  query_mapping, query_mapping2,end_mapping]:
                mapping.zero_()   
           
            is_support.fill_(0)
            

            for i in range(len(cur_batch)):    
                case = cur_batch[i]
                query_length = len(case.query_input_ids)
                # print(f'all_doc_tokens is {case.doc_tokens}')
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))
                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1  #找到query的标记
                    query_mapping2[i, j] = 1
                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = self.max_length
                    elif case.end_position[0] < self.max_length:
                        y1[i] = case.start_position[0]   
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = self.max_length
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = self.max_length
                    y2[i] = self.max_length
                    q_type[i] = 1  
                elif case.ans_type == 2:
                    y1[i] = self.max_length+1
                    y2[i] = self.max_length+1
                    q_type[i] = 2
                elif case.ans_type == 3:
                    y1[i] = self.max_length+2
                    y2[i] = self.max_length+2
                    q_type[i] = 3

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):   
                    is_sp_flag = j in case.sup_fact_ids   
                    start, end = sent_span

                    if start < end:

                        start_position[i][j] = start - query_length
                        end_position[i][j] = end - query_length
                        is_support[i, j] = int(is_sp_flag)   
                        all_mapping[i, start:end+1, j] = 1   
                        start_mapping[i, j, start] = 1
                        end_mapping[i,j,end] = 1




                ids.append(case.qas_id)
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))
                start_position[i] = start_position[i][:max_sent_cnt]
                end_position[i] = end_position[i][:max_sent_cnt]
            # input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            # max_c_len = int(input_lengths.max())  #变长处理
            # print(max_c_len)
            # print()
            max_c_len = 512
            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len+3].contiguous(),
                'query_mapping2': query_mapping2[:cur_bsz, :max_c_len].contiguous(),
                'y1': y1[:cur_bsz].to("cuda"),
                'y2': y2[:cur_bsz].to("cuda"),
                'ids': ids,
                'q_type': q_type[:cur_bsz].to("cuda"),
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len].to("cuda"),
                'end_mapping': end_mapping[:cur_bsz, :max_sent_cnt, :max_c_len].to("cuda"),
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt].to("cuda"),
                'is_support':is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_position':start_position,
                'end_position':end_position,
                'query_length':query_length,
            }
