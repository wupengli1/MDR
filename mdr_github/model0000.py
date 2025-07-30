
import torch
import torch.nn as nn
from sentence_transformers.util import cos_sim

from Encoder import Encoder
from Decoder import Decoder

import pytorch_lightning as pl

# from MemNet import UserMemory
from MemNet1 import UserMemory as UserMemory1

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
        self.encoder1 = Encoder()
        self.decoder = Decoder()

        # 除以温度系数temp
        self.cos_sim = Similarity(temp=1)
        self.learn_p = nn.Parameter(torch.normal(0,0.1,size=(1,config.embedding_dim))).cuda()

        self.learn_u = nn.Parameter(torch.normal(0,0.1,size=(1,config.embedding_dim))).cuda()
        # decoder
        self.user_memnet = UserMemory1(vocab_size=5,
                                      query_size=config.embedding_dim, memory_size=config.embedding_dim,
                                      max_hop=4,
                                      dropout=0.1, init_std=0.02,
                                      padding_idx=None)
        self.user_memnet1 = UserMemory1(vocab_size=5,
                                      query_size=config.embedding_dim, memory_size=config.embedding_dim ,
                                      max_hop=4,
                                      dropout=0.1, init_std=0.02,
                                      padding_idx=None)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def forward(self, inputs, is_train=False):
        text_posts = inputs['posts']  # [batch, seq]
        text_his = inputs['his']  # [batch, seq]
        text_responses = inputs['responses']  # [batch, seq]
        text_persons = inputs['persons']  # [batch, seq]
        text_posts_responses = inputs['p_r']  # only for train
        # x_t_t = self.encoder(text_responses)

        # state: [batch, seq ,dim]
        lastre = True
        for hisbatch in range(len(text_his)):
            # if is_train:
            x_start_response_gpt = self.encoder([text_responses[hisbatch]])
            x_start_response_gpt1 = self.encoder1([text_responses[hisbatch]])

            x_start_posts_gpt_fix = self.encoder([text_posts[hisbatch]])
            x_start_posts_gpt = self.encoder1([text_posts[hisbatch]])

            text_his_len = len(text_his[hisbatch])
            person_list = [person_i + '.' for person_i in text_persons[hisbatch].split('.')[:-1]]

            state_persons_gpt = self.encoder1(person_list)  # .detach()
            state_persons_gpt_fix = self.encoder(person_list)  # .detach()

            for hisi in range(0, text_his_len):
                str_cur = [text_his[hisbatch][hisi]]
                state_hisr_gpt = self.encoder(str_cur)
                if hisi == 0:
                    state_hisr_list_gpt = state_hisr_gpt
                else:
                    state_hisr_list_gpt = torch.concat((state_hisr_list_gpt, state_hisr_gpt), 0)

            posts = torch.cat((x_start_posts_gpt, x_start_posts_gpt_fix), 0)
            profile = torch.cat((state_persons_gpt, state_persons_gpt_fix), 0)
            if text_his_len == 0:
                user_mem = self.user_memnet.load_memory(self.learn_p)
                profile_res = self.user_memnet(
                    context=posts,
                    user_memory_list=user_mem,
                    profile=profile)
                user_mem1 = self.user_memnet1.load_memory(self.learn_u)
                res2 = self.user_memnet1(
                    context=posts,
                    user_memory_list=user_mem1,
                    profile=profile_res)
            else:
                str_cur1 = [text_his[hisbatch][-1]]
                state_hisr_list = self.encoder1(str_cur1)
                state_hisr_list_gpt_cur1 = state_hisr_list_gpt[-1].unsqueeze(0)
                if state_hisr_list_gpt.shape[0] < 4:
                    for len_his in range(state_hisr_list_gpt.shape[0] - 1):
                        str_cur = [text_his[hisbatch][len_his]]
                        state_hisr_list_gpt_cur1 = torch.cat((state_hisr_list_gpt[len_his].unsqueeze(0),
                                                              state_hisr_list_gpt_cur1),
                                                             dim=0)
                        state_hisr_list = torch.cat((self.encoder1(str_cur), state_hisr_list), dim=0)
                else:
                    # a = self.cos_sim(x_start_posts_gpt_fix, state_hisr_list_gpt[:-1])
                    sorce_hiss = torch.topk(self.cos_sim(x_start_posts_gpt_fix, state_hisr_list_gpt[:-1]), 3, )
                    for sorce_his in sorce_hiss[1]:
                        str_cur = [text_his[hisbatch][sorce_his]]
                        state_hisr_list_gpt_cur1 = torch.cat((state_hisr_list_gpt[int(sorce_his)].unsqueeze(0),
                                                              state_hisr_list_gpt_cur1),
                                                             dim=0)
                        state_hisr_list = torch.cat((self.encoder1(str_cur), state_hisr_list), dim=0)
                context = torch.cat((state_hisr_list, state_hisr_list_gpt_cur1), 0)
                user_mem = self.user_memnet.load_memory(self.learn_p)
                profile_res = self.user_memnet(
                    context=posts,
                    user_memory_list=user_mem,
                    profile=torch.cat((profile, context), 0))
                user_mem1 = self.user_memnet1.load_memory(self.learn_u)
                res2 = self.user_memnet1(
                    context=posts,
                    user_memory_list=user_mem1,
                    profile=torch.cat((profile_res, context), 0))

            if not is_train:
                if lastre:
                    x_t_t = res2
                    lastre = False
                else:
                    x_t_t = torch.concat((x_t_t, res2))
        return x_t_t

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
