import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertEncoder

VERY_NEGATIVE_NUMBER = -1e30


class CailModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CailModel, self).__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size *4
        # self.attention = MultiLinearLayer(2, config.hidden_size, 1)
        # self.attention = nn.Linear(self.hidden_size, 1)

        # self.qa_outputs = nn.Linear(self.hidden_size * 4, 2)
        # self.retionale_outputs = nn.Linear(self.hidden_size * 4, 1)
        # self.unk_ouputs = nn.Linear(self.hidden_size, 1)
        # self.doc_att = nn.Linear(self.hidden_size * 4, 1)
        # self.yes_no_ouputs = nn.Linear(self.hidden_size * 4, 2)
        self.qa_outputs = MultiLinearLayer(2, config.hidden_size*4, 2)
        self.retionale_outputs = MultiLinearLayer(2, config.hidden_size*4,1)
        # self.retionale_outputs2 = MultiLinearLayer(2, config.hidden_size * 4, 1)
        # self.unk_ouputs1 = MultiLinearLayer(2, config.hidden_size * 4, 1)
        # self.doc_att = MultiLinearLayer(2, config.hidden_size*4, 1)
        # self.yes_no_ouputs = MultiLinearLayer(2, config.hidden_size*4, 2)
        self.ouputs_cls_3 = MultiLinearLayer(2, config.hidden_size * 4, 3)
        # self.sp_linear = nn.Linear(self.hidden_size, 1)
        # self.sp_linear1 = nn.Linear(self.hidden_size, 1)
        # self.cross_passage_encoder = CrossPassageEncoder.from_pretrained('/data/wuyunzhao/pre-model/chinese_roberta_wwm_ext_pytorch')
        # self.fc_2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.fc_sf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        # self.n_layers = 2
        # self.multiLinearLayer_rationale_logits = MultiLinearLayer(self.n_layers, self.hidden_size *4, 1)
        # self.multiLinearLayer_unk_logits = MultiLinearLayer(self.n_layers, self.hidden_size, 1)
        # self.multiLinearLayer_attention = MultiLinearLayer(self.n_layers, self.hidden_size*4, 1)
        # self.multiLinearLayer_answer_logits = MultiLinearLayer(self.n_layers, self.hidden_size*4, 2)
        # self.multiLinearLayer_sp_logits = MultiLinearLayer(self.n_layers, self.hidden_size*4, 1)
        # self.multiLinearLayer_yes_no_logits = MultiLinearLayer(self.n_layers, self.hidden_size*4, 2)
        self.max_query_length = 64
        self.beta = 100
        # self.input_dim = config.input_dim

        self.init_weights()
        # self.start_linear = nn.Linear(self.input_dim, 1)
        # self.end_linear = nn.Linear(self.input_dim, 1)
        #
        # self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans
        #
        # self.cache_S = 0
        # self.cache_mask = None

    @staticmethod
    def mean_pooling(input, mask):
        # print(mask.sum(dim=1))
        mask = mask.sum(dim=1)[:,:,None] + VERY_NEGATIVE_NUMBER
        mean_pooled = input.sum(dim=1) / mask
        return mean_pooled

    @staticmethod
    def compute_loss(batch, sp_logits):
        sp_loss_fct = BCEWithLogitsLoss(reduction='none')

        sent_num_in_batch = batch["start_mapping"].sum()
        loss = 10 * sp_loss_fct(sp_logits.view(-1),
                                batch['is_support'].float().contiguous().view(-1)).sum() / sent_num_in_batch
        return loss

    def get_output_mask(self, outer):  # 挡住一些位置
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 128)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch):

        input_ids = batch['context_idxs']
        token_type_ids = batch['segment_idxs']
        attention_mask = batch['context_mask']
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1
        start_positions = batch['y1']
        end_positions = batch['y2']
        query_mapping = batch['query_mapping2']
        qlen = batch['query_length']
        sent_start = batch['start_position']
        sent_end = batch['end_position']
        _, pooled_output, sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = torch.cat((sequence_output[-4], sequence_output[-3], sequence_output[-2],
                                     sequence_output[-1]), -1)

        batch_size = sequence_output.size(0)
        seq_length = sequence_output.size(1)
        hidden_size = sequence_output.size(2)

        # batch_size = sequence_output[-1].size(0)
        # seq_length = sequence_output[-1].size(1)
        # hidden_size = sequence_output[-1].size(2)
        # eij1 = self.attention(sequence_output[-4].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij2 = self.attention(sequence_output[-3].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij3 = self.attention(sequence_output[-2].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij4 = self.attention(sequence_output[-1].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij = torch.stack([eij1, eij2, eij3, eij4], 2)
        # a = F.tanh(eij)
        # a = F.softmax(a, 2)
        # sequence_output = a[:, :, 0:1] * sequence_output[-4] + a[:, :, 1:2] * sequence_output[-3] + a[:, :, 2:3] * \
        #                   sequence_output[-2] + a[:, :, 3:4] * sequence_output[-1]

        sequence_output_matrix = sequence_output.view(batch_size * seq_length, hidden_size)
        rationale_logits = self.retionale_outputs(sequence_output_matrix) # 集中在某些片段
        rationale_logits = F.softmax(rationale_logits)

        rationale_logits = rationale_logits.view(batch_size, seq_length)
        # rationale_logits2 = self.retionale_outputs2(sequence_output_matrix)  # 集中在某些片段
        # rationale_logits2 = F.softmax(rationale_logits2)
        #
        # rationale_logits2 = rationale_logits2.view(batch_size, seq_length)
        # sp_hidden = sequence_output * rationale_logits2.unsqueeze(2)
        final_hidden = sequence_output * rationale_logits.unsqueeze(2)
        sequence_output = final_hidden.view(batch_size * seq_length, hidden_size)

        logits = self.qa_outputs(sequence_output).view(batch_size, seq_length, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(2) - 1e30 * (1 - context_mask)  # mask掉被遮住的logits
        # end_logits = end_logits.squeeze(2) - 1e30 * (1 - context_mask)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # [000,11111] 1代表了文章
        # [batch, seq_len] [batch, seq_len]
        rationale_logits = rationale_logits * attention_mask.float()
        # [batch, seq_len, 1] [batch, seq_len]
        start_logits = start_logits * rationale_logits
        end_logits = end_logits * rationale_logits
        # unk
        # unk_logits = self.unk_ouputs(pooled_output)
        unk_logits = torch.max(final_hidden, 1)[0]
        unk_yes_no_logits = self.ouputs_cls_3(unk_logits)
        unk_logits, yes_logits, no_logits = unk_yes_no_logits.split(1, dim=-1)
        # doc_attn
        # attention = self.doc_att(sequence_output)
        # attention = attention.view(batch_size, seq_length)
        # attention = attention * token_type_ids.float() + (1 - token_type_ids.float()) * VERY_NEGATIVE_NUMBER
        #
        # attention = F.softmax(attention, 1)
        # attention = attention.unsqueeze(2)
        # attention_pooled_output = attention * final_hidden
        # final_hidden = attention_pooled_output + final_hidden
        # unk_logits = torch.max(final_hidden,1)[0]
        # unk_logits = self.unk_ouputs1(unk_logits)
        # attention_pooled_output = attention_pooled_output.sum(1)
        #
        # yes_no_logits = self.yes_no_ouputs(attention_pooled_output)
        # yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

        new_start_logits = torch.cat([start_logits, unk_logits, yes_logits, no_logits], 1)
        new_end_logits = torch.cat([end_logits, unk_logits, yes_logits, no_logits], 1)
        # final_hidden = self.fc_2(final_hidden)
        # sp_hidden += final_hidden
        sp_state = all_mapping.unsqueeze(3) * final_hidden.unsqueeze(2)  # N x 512 x 100 x 768  将最后两维复制到同一维度
        # mean_state = self.mean_pooling(sp_state, all_mapping)
        max_state = sp_state.max(1)[0]
        # max_state = sp_state.sum(1)[0] / (all_mapping.sum(1)[:,:,None] + VERY_NEGATIVE_NUMBER)
        # sp_state = torch.cat([mean_state,max_state],-1)
        sp_logits = self.fc_sf(max_state)  # N 100
        # sp_logits = self.sp_linear1(all_tok_vecs)
        if len(sp_logits.size()) > 1:
            sp_logits = sp_logits.squeeze(-1)
        # If we are on multi-GPU, split add a dimension
        sp_loss = self.compute_loss(batch, sp_logits)
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = new_start_logits.size(1)
        start_positions.clamp_(1, ignored_index)
        end_positions.clamp_(1, ignored_index)
        # print(start_positions)
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(new_start_logits, start_positions)
        end_loss = loss_fct(new_end_logits, end_positions)
        # print(start_loss)
        rationale_positions = token_type_ids.float()
        alpha = 0.25
        gamma = 2.
        rationale_loss = -alpha * ((1 - rationale_logits) ** gamma) * rationale_positions * torch.log(
            rationale_logits + 1e-8) - (1 - alpha) * (rationale_logits ** gamma) * (
                                 1 - rationale_positions) * torch.log(1 - rationale_logits + 1e-8)
        rationale_loss = (rationale_loss * token_type_ids.float()).sum() / token_type_ids.float().sum()

        total_loss = (start_loss + end_loss) / 2 + rationale_loss * self.beta + sp_loss
        # total_loss = (start_loss + end_loss) / 2

        return total_loss, (start_loss + end_loss) / 2, sp_loss, new_start_logits, new_end_logits, sp_logits
        #                                     # N 512 1 768

class CrossPassageEncoder(BertPreTrainedModel):
    """
    When the individual chunk encodings are built from PassageEncoder, we assemble them all together to capture
    cross passage interactions.
    """
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.init_weights()
        self.num_hidden_layers = config.num_hidden_layers
    def forward(self, vectors_in):
        # vectors_in is expected to be free of padding
        extended_attention_mask = torch.zeros(vectors_in.shape[0], 1,1,vectors_in.shape[1],
                                              device=vectors_in.device, dtype=vectors_in.dtype)
        head_mask = [None] * self.num_hidden_layers
        encoded_layers = self.encoder(
            vectors_in, extended_attention_mask, head_mask = head_mask,encoder_hidden_states=False, encoder_attention_mask=False
        )

        encoded_layers = encoded_layers[0]
        return encoded_layers


class MultiLinearLayer(nn.Module):
    def __init__(self, layers, hidden_size, output_size, activation=None):
        super(MultiLinearLayer, self).__init__()
        self.net = nn.Sequential()

        for i in range(layers - 1):
            self.net.add_module(str(i) + 'linear', nn.Linear(hidden_size, hidden_size))
            self.net.add_module(str(i) + 'relu', nn.ReLU(inplace=True))

        self.net.add_module('linear', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)
