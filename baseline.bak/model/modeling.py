import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertEncoder


# config1 = BertConfig.from_pretrained("/data/wuyunzhao/pre-model/chinese_roberta_wwm_ext_pytorch")

class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim *4

        # self.sp_linear = nn.Linear(self.input_dim, 1)
        # self.start_linear = nn.Linear(self.input_dim, 1)
        # self.end_linear = nn.Linear(self.input_dim, 1)
        #
        # self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans

        self.cache_S = 0
        self.cache_mask = None

        # self.cross_passage_encoder = CrossPassageEncoder(config1)
        # self.fc = nn.Linear(1536 , 1)
        # self.fc2 = nn.Linear(self.input_dim,768)
        self.fc_sf = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, 1),
        )
        # self.lstm = nn.LSTM(self.input_dim, self.input_dim, batch_first=True)

    @staticmethod
    def mean_pooling(input, mask):
        mean_pooled = input.sum(dim=1) / (mask.sum(dim=1).unsqueeze(-1)+(1e-30))
        return mean_pooled

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        # if not hasattr(self, '_flattened'):
        #     self.lstm.flatten_parameters()
        #     setattr(self, '_flattened', True)
        pooled = batch['pooled']
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        qlen = batch['query_length']
        sent_start = batch['start_position']
        sent_end = batch['end_position']
        # query = input_state * query_mapping[:,:,None]
        # query = query.max(1)[0]
        # query = query[:,None,:].repeat(1, all_mapping.size()[2],1)
        # # pooled = pooled[:,None,:].repeat(1, all_mapping.size()[2],1)
        # print(query.size())
        # print(all_mapping)
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  # N x 512 x 100 x 300
        sp_state = sp_state.max(1)[0]
        sp_logits = self.fc_sf(sp_state)
        # sp_state = torch.cat([query[:,None,:], sp_state], 1)
        # sp_state = self.fc(sp_state)z
        # h, c = self.lstm(sp_state,None)
        # sp_state = self.fc2(sp_state)

        # print(sp_state.size())

        # sp_state = self.cross_passage_encoder(sp_state)

        # input_state = self.fc2(input_state[:,qlen:,:])
        # all_tok_vecs = self.cross_passage_encoder(input_state)
        # flag = True
        # for i in range(all_tok_vecs.size()[0]):
        #     if flag:
        #         start_vectors = all_tok_vecs[:, sent_start[i]]
        #         end_vectors = all_tok_vecs[:, sent_end[i]]
        # start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)
        # sp_logits = self.fc(start_end_concatenated)




        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)

        return sp_logits
class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """

    def __init__(self, config, encoder):
        super(BertSupportNet, self).__init__()
        # self.bert_model = BertModel.from_pretrained(config.bert_model)
        self.encoder = encoder
        self.graph_fusion_net = SupportNet(config)
        # self.attention = nn.Linear(config.input_dim, 1)
    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        # roberta不可以输入token_type_ids

        _, pooled, sequence_output= self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,#可以注释
                                              attention_mask=doc_mask)
        sequence_output = torch.cat((sequence_output[-4], sequence_output[-3], sequence_output[-2],
                                     sequence_output[-1]), -1)
        # batch_size = sequence_output[-1].size(0)
        # seq_length = sequence_output[-1].size(1)
        # hidden_size = sequence_output[-1].size(2)
        # eij1 = self.attention(sequence_output[-4].view(-1 ,hidden_size)).view(batch_size, seq_length)
        # eij2 = self.attention(sequence_output[-3].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij3 = self.attention(sequence_output[-2].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij4 = self.attention(sequence_output[-1].view(-1, hidden_size)).view(batch_size, seq_length)
        # eij = torch.stack([eij1, eij2, eij3, eij4], 2)
        # a = F.tanh(eij)
        # a = F.softmax(a,2)
        # sequence_output = a[:,:,0:1] * sequence_output[-4] + a[:,:,1:2] * sequence_output[-3] + a[:,:,2:3] * sequence_output[-2] + a[:,:,3:4] * sequence_output[-1]
        batch['context_encoding'] = sequence_output
        batch['pooled'] = pooled
        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """

    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  # 就是args
        # self.n_layers = config.n_layers  # 2
        self.max_query_length = 50
        self.prediction_layer = SimplePredictionLayer(config)

    def forward(self, batch, debug=False):
        context_encoding = batch['context_encoding']
        predictions = self.prediction_layer(batch, context_encoding)

        sp_logits = predictions

        return sp_logits

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
        extended_attention_mask = torch.zeros(vectors_in.shape[0],1,1,vectors_in.shape[1],
                                              device=vectors_in.device, dtype=vectors_in.dtype)
        head_mask = [None] * self.num_hidden_layers
        encoded_layers = self.encoder(
            vectors_in, extended_attention_mask, head_mask = head_mask,encoder_hidden_states=False, encoder_attention_mask=False
        )
        # print(len(encoded_layers))
        encoded_layers = encoded_layers[-1]
        return encoded_layers