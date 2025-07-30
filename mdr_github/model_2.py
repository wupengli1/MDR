
import torch
import torch.nn as nn
from sentence_transformers.util import cos_sim

from Encoder import Encoder
from Decoder_post import Decoder
import pytorch_lightning as pl

# from MemNet import UserMemory
from MemNet1 import UserMemory as UserMemory1
import math
class Similarity(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MLModel(pl.LightningModule):
    def __init__(self, config):
        super(MLModel, self).__init__()
        self.config = config

        # Encoder personality, reply, context are essentially text, so share a coder
        self.encoder = Encoder()
        # self.encoder1 = Encoder()
        self.decoder = Decoder()

        # 除以温度系数temp
        self.cos_sim = Similarity(temp=1)
        # self.learn_p = nn.Parameter(torch.normal(0,0.1,size=(1,config.embedding_dim))).cuda()
        #
        # self.learn_u = nn.Parameter(torch.normal(0,0.1,size=(1,config.embedding_dim))).cuda()
        # decoder
        # self.user_memnet = UserMemory1(vocab_size=5,
        #                               query_size=config.embedding_dim, memory_size=config.embedding_dim,
        #                               max_hop=4,
        #                               dropout=0.1, init_std=0.02,
        #                               padding_idx=None)
        # self.user_memnet1 = UserMemory1(vocab_size=5,
        #                               query_size=config.embedding_dim, memory_size=config.embedding_dim ,
        #                               max_hop=4,
        #                               dropout=0.1, init_std=0.02,
        #                               padding_idx=None)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        # for name, param in self.decoder.named_parameters():
        #     param.requires_grad = False

    def forward(self, inputs, is_train=False):
        text_posts = inputs['posts']  # [batch, seq]
        # text_his = inputs['his']  # [batch, seq]
        text_responses = inputs['responses']  # [batch, seq]
        text_persons = inputs['persons']  # [batch, seq]
        text_posts_responses = inputs['p_r']  # only for train
        x_t_t = inputs['features'][0].unsqueeze(0)  # only for train
        for i in range(1, len(inputs['features'])):
            x_t_t = torch.concat((x_t_t, inputs['features'][i].unsqueeze(0)), 0)
        x_t_true = inputs['features_true'][0].unsqueeze(0)  # only for train
        for i in range(1, len(inputs['features_true'])):
            x_t_true = torch.concat((x_t_true, inputs['features_true'][i].unsqueeze(0)), 0)
        sig = torch.Tensor([0.5]).cuda()
        pi = torch.Tensor([math.pi]).cuda()
        for hisbatch in range(len(text_posts)):
            if is_train:
                # a = self.encoder.tok.tokenize(text_responses[hisbatch])
                # b = self.encoder.tok.tokenize(text_posts[hisbatch])
                # c = self.encoder.tok.tokenize(text_posts_responses[hisbatch]) .split('<|endoftext|>')[0]
                gpt2_tok = self.encoder.tok.tokenize(text_responses[hisbatch])
                batch_input_encode={}
                len_post = len(self.decoder.tok.tokenize(text_posts[hisbatch]))

                for i in range(len(gpt2_tok)-1):
                    temp = self.encoder.tok.convert_tokens_to_ids(
                        gpt2_tok[i])
                    bili = [[temp,50256]]
                    bili = torch.IntTensor(bili).cuda()
                    outputs = self.encoder.model(
                        input_ids=bili,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    response_embed = outputs.hidden_states[-1][:, -1, :]  # [batch, dim]
                    if i ==0:
                        sorce_response_embed = self.cos_sim(
                            x_t_true[hisbatch].unsqueeze(0).repeat_interleave(len(response_embed), 0),
                            response_embed)
                    else:
                        sorce_response_embed = torch.cat((sorce_response_embed, self.cos_sim(
                            x_t_true[hisbatch].unsqueeze(0).repeat_interleave(len(response_embed), 0),
                            response_embed)),0)
                u = torch.max(sorce_response_embed)
                zhongjian = torch.exp(-(sorce_response_embed - u) ** 2 / (2 * sig ** 2)) / (torch.sqrt(2 * pi) * sig)
                jiewei = torch.max(zhongjian).unsqueeze(0)

                jieguo = torch.cat(
                    (torch.zeros((len_post)).cuda(),
                     zhongjian, jiewei,), 0)

                if hisbatch == 0:

                    if len(jieguo) <= self.decoder.max_length:
                        sim_score = [jieguo]
                    else:
                        sim_score = [jieguo[:self.decoder.max_length]]
                else:
                    if len(jieguo) <= self.decoder.max_length:
                        sim_score.append(jieguo)
                    else:
                        sim_score.append(jieguo[:self.decoder.max_length])
            else:
                pass

        if is_train:

            decode_loss = self.decoder(is_train, text_persons, text_posts, text_responses, text_posts_responses, x_t_t,
                                       None, sim_score)
            return decode_loss
        else:
            decode_loss = self.decoder(is_train, text_persons, text_posts, text_responses, text_posts_responses, x_t_t,
                                       None, None)
            return decode_loss
    def print_parameters(self):
        r""" Statistical parameter """
        total_num = 0
        for param in self.parameters():
            num = 1
            if param.requires_grad:
                size = param.size()
                for dim in size:
                    num *= dim
            total_num += num
        print(f"Total number of parameters: {total_num}")


