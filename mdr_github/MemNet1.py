# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# class AttrProxy(object):
#     """
#     Translates index lookups into attribute lookups.
#     To implement some trick which able to use list of nn.Module in a nn.Module
#     see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
#     """
#     def __init__(self, module, prefix):
#         self.module = module
#         self.prefix = prefix
#
#     def __getitem__(self, i):
#         return getattr(self.module, self.prefix + str(i))


class UserMemory(nn.Module):
    """
    User Memory Network.
    """

    def __init__(
            self,
            vocab_size,
            query_size,
            memory_size,
            max_hop=1,
            dropout=0.1,
            init_std=0.02,
            padding_idx=None,
            mode="general",
    ):
        super(UserMemory, self).__init__()
        assert (mode in ["general", "mlp"]), ("Unsupported attention mode: {mode}")
        self.vocab_size = vocab_size
        self.query_size = query_size
        self.memory_size = memory_size
        self.max_hop = max_hop
        self.init_std = init_std
        self.padding_idx = padding_idx
        self.mode = mode

        if self.mode == "general":
            self.linear_query = nn.ModuleList(
                [nn.Linear(self.query_size, self.memory_size, bias=False) for _ in range(self.max_hop)])
            self.linear_query1 = nn.ModuleList(
                [nn.Linear(self.query_size, self.memory_size, bias=False) for _ in range(self.max_hop)])
        elif self.mode == "mlp":
            self.linear_query = nn.ModuleList(
                [nn.Linear(self.query_size, self.memory_size, bias=True) for _ in range(self.max_hop)])
            self.linear_memory = nn.ModuleList(
                [nn.Linear(self.memory_size, self.memory_size, bias=False) for _ in range(self.max_hop)])
            self.v = nn.ModuleList([nn.Linear(self.memory_size, 1, bias=False) for _ in range(self.max_hop)])
            self.tanh = nn.Tanh()

        # for hop in range(self.max_hop + 1):
        #     U = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.memory_size, padding_idx=self.padding_idx)
        #     U.weight.data.normal_(0, 0.1)
        #     self.add_module("U_{}".format(hop), U)
        # self.U = AttrProxy(self, "U_")

        if dropout > 0 and dropout < 1:
            self.dropout_layer = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        for hop in range(self.max_hop):
            self.linear_query[hop].weight.data.normal_(mean=0.0, std=self.init_std)
            if self.linear_query[hop].bias is not None:
                self.linear_query[hop].bias.data.zero_()

    def load_memory(self, user_profile):
        """
        Args:
            user_profile: (batch_size, profile_len)
            context: (batch_size, context_len, hidden_size)
        """
        memory_list = []
        for hop in range(self.max_hop):
            embed_state = user_profile
            # embed_state_next = user_profile
            memory_list.append(embed_state)
        # memory_list.append(embed_state_next)
        return memory_list

    def memory_address(self, profile, context, hop, mask=None):
        if self.mode == "general":
            # assert self.memory_size == profile.size(-1)
            key = self.linear_query[hop](profile)
            attn = torch.matmul(key, context.transpose(0, 1))
        else:
            hidden_sum = self.linear_query[hop](profile) + self.linear_memory[hop](context)
            key = self.tanh(hidden_sum)
            attn = self.v[hop](key).squeeze(-1)

        if mask is not None:
            attn.masked_fill_(mask.eq(0), -float("inf"))
        weights = self.softmax(attn)
        return weights
    def memory_address1(self, profile, context, hop, mask=None):
        if self.mode == "general":
            # assert self.memory_size == profile.size(-1)
            key = self.linear_query1[hop](profile)
            attn = torch.matmul(key, context.transpose(0, 1))
        else:
            hidden_sum = self.linear_query1[hop](profile) + self.linear_memory[hop](context)
            key = self.tanh(hidden_sum)
            attn = self.v[hop](key).squeeze(-1)

        if mask is not None:
            attn.masked_fill_(mask.eq(0), -float("inf"))
        weights = self.softmax(attn)
        return weights

    def forward(self, context, user_memory_list, mask=None,profile=None):
        """
        Update user memory.
        Args:
            context: (batch_size, context_len, hidden_size)
            mask: (batch_size, context_len)
        Returns:
            final_memory: (batch_size, profile_len, hidden_size)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, user_memory_list[0].size(1), 1)

        final_memory = None
        for hop in range(self.max_hop):
            u_k = user_memory_list[hop]
            attn_weights = self.memory_address(u_k, context, hop, mask=mask)
            o_k1 = torch.matmul(attn_weights, context)
            attn_weights1 = self.memory_address1(o_k1, profile, hop, mask=mask)
            o_k = torch.matmul(attn_weights1, profile)
            if hop + 1 == self.max_hop:
                final_memory = u_k + o_k
                break
            else:
                user_memory_list[hop + 1] = u_k + o_k

        assert final_memory is not None
        return final_memory
