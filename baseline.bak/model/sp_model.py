import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from transformers import BertPreTrainedModel, BertConfig
from transformers.modeling_bert import BertEncoder

config1 = BertConfig.from_pretrained("/data/wuyunzhao/pre-model/chinese_roberta_wwm_ext_pytorch")
class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)

        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans

        self.cache_S = 0
        self.cache_mask = None

        self.cross_passage_encoder = CrossPassageEncoder(config1)
        self.fc = nn.Linear(self.input_dim , 1)
        self.fc_sf = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, 1),
        )
        self.hidden = config.hidden

        self.rnn = EncoderRNN(config.char_hidden + self.word_dim, config.hidden, 1, True, True, 1 - config.keep_prob,
                              False)

        self.qc_att = BiAttention(config.hidden * 2, 1 - config.keep_prob)
        self.linear_1 = nn.Sequential(
            nn.Linear(config.hidden * 8, config.hidden),
            nn.ReLU()
        )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1 - config.keep_prob, False)
        self.self_att = BiAttention(config.hidden * 2, 1 - config.keep_prob)
        self.linear_2 = nn.Sequential(
            nn.Linear(config.hidden * 8, config.hidden),
            nn.ReLU()
        )

        self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1 - config.keep_prob, False)
        self.linear_sp = nn.Linear(config.hidden * 2, 1)

        self.rnn_start = EncoderRNN(config.hidden * 3, config.hidden, 1, False, True, 1 - config.keep_prob, False)
        self.linear_start = nn.Linear(config.hidden * 2, 1)

        self.rnn_end = EncoderRNN(config.hidden * 3, config.hidden, 1, False, True, 1 - config.keep_prob, False)
        self.linear_end = nn.Linear(config.hidden * 2, 1)

        self.rnn_type = EncoderRNN(config.hidden * 3, config.hidden, 1, False, True, 1 - config.keep_prob, False)
        self.linear_type = nn.Linear(config.hidden * 2, 3)

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
        pooled = batch['pooled']
        query_mapping = batch['query_mapping']  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask']  # bert里实际有输入的位置
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 每个句子的token对应为1

        qlen = batch['query_length']
        sent_start = batch['start_position']
        sent_end = batch['end_position']


        # sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)  # N x sent x 512 x 300
        #
        # sp_state = sp_state.max(1)[0]
        # #
        # sp_logits = self.sp_linear(sp_state)

        all_tok_vecs = self.cross_passage_encoder(input_state[:,qlen:,:])
        flag = True
        for i in range(all_tok_vecs.size()[0]):
            if flag:
                start_vectors = all_tok_vecs[:, sent_start[i]]
                end_vectors = all_tok_vecs[:, sent_end[i]]
        start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)
        sp_logits = self.fc_sf(start_end_concatenated)




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

    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'], batch['context_mask'], batch['segment_idxs']
        # roberta不可以输入token_type_ids
        all_doc_encoder_layers, pooled= self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,#可以注释
                                              attention_mask=doc_mask)
        batch['context_encoding'] = all_doc_encoder_layers
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
        extended_attention_mask = torch.zeros(vectors_in.shape[0], 1,1,vectors_in.shape[1],
                                              device=vectors_in.device, dtype=vectors_in.dtype)
        head_mask = [None] * self.num_hidden_layers
        encoded_layers = self.encoder(
            vectors_in, extended_attention_mask, head_mask = head_mask,encoder_hidden_states=False, encoder_attention_mask=False
        )

        encoded_layers = encoded_layers[0]
        return encoded_layers

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))