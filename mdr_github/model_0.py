
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
import pytorch_lightning as pl

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
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs, is_train=False):
        text_posts = inputs['posts']  # [batch, seq]
        text_responses = inputs['responses']  # [batch, seq]
        text_persons = inputs['persons']  # [batch, seq]
        text_posts_responses = inputs['p_r']  # only for train

        response_state = self.encoder(text_responses)

        if is_train:
            decode_loss = self.decoder(is_train, text_persons, text_posts, text_responses, text_posts_responses, response_state,
                                       None,None)
            return  decode_loss
        else:
            decode_loss = self.decoder(is_train, text_persons, text_posts, text_responses, text_posts_responses,response_state,
                                       None,None)
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

