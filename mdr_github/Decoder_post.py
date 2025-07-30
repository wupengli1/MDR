
from transformers import GPT2Tokenizer
from modeling_gpt2_1 import GPT2LMHeadModel
import torch
from config import Config
from transformers import BertTokenizer
import torch.nn.functional as F
import pytorch_lightning as pl


class Decoder(pl.LightningModule):
    r""" decoder """

    def __init__(self):  # dropout
        super(Decoder, self).__init__()
        config = Config()
        data_language = config.data_language

        if data_language == 'EN':
            self.tok = GPT2Tokenizer.from_pretrained("F:/premodel/gpt2", do_lower_case=False)
            self.model = GPT2LMHeadModel.from_pretrained("F:/premodel/gpt2")
            self.tok.pad_token = self.tok.eos_token
        else:
            self.tok = BertTokenizer.from_pretrained(
                "F:/premodel/gpt2chinesecluecorpussmall")  # uer/gpt2-chinese-cluecorpussmall
            self.model = GPT2LMHeadModel.from_pretrained("F:/premodel/gpt2chinesecluecorpussmall",
                                                         ignore_mismatched_sizes=True)

        self.config = config
        self.max_length = self.config.max_len
        print('eos_token:', self.tok.eos_token_id, self.tok.eos_token)
        print('pad_token:', self.tok.pad_token_id, self.tok.pad_token)
        print('unk_token:', self.tok.unk_token_id, self.tok.unk_token)

    def forward(self, if_train, p, x, y, x_y, z_r, z_p,sim_score):  # The initial state [layers*directions, batch, dim]
        def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
            """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                Args:
                    logits: logits distribution shape (vocab size)
                    top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                    top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            """
            assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                # torch.topk()Returns the last d maximum top_k element, the return value for two-dimensional (values, indices)
                # Said other dimensions by computer to inference
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value  # Other elements outside of topk logits a value minus infinity

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # Diminishing logits for sorting
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = filter_value
            return logits

        z = z_r#  + z_p
        if if_train:
            batch_input_encode_x_y = self.tok.batch_encode_plus(
                x_y,
                max_length=self.max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            batch_input_encode_x = self.tok.batch_encode_plus(
                x,
                max_length=self.max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            batch_input_encode_x_y['labels'] = torch.tensor([
                [-100 if mask == 0 else token for mask, token in mask_and_tokens]
                for mask_and_tokens in [zip(masks, labels) for masks, labels in
                                        zip(batch_input_encode_x_y['attention_mask'],
                                            batch_input_encode_x_y['input_ids'])]])

            # By filling 0 two matrices the same size
            try:
                batch_input_encode_x['attention_mask'] = torch.cat((batch_input_encode_x['attention_mask'][:, 1:],
                                                                    torch.zeros((batch_input_encode_x[
                                                                                     'attention_mask'].shape[0],
                                                                                 batch_input_encode_x_y[
                                                                                     'attention_mask'].shape[1] -
                                                                                 batch_input_encode_x[
                                                                                     'attention_mask'].shape[1] + 1))),
                                                                   dim=1)
            except:
                print('x_y', x_y)
                print('x', x)
            batch_input_encode_x_y['labels'] = torch.where(batch_input_encode_x['attention_mask'] == 1, -100,
                                                           batch_input_encode_x_y['labels'])
            # Tectonic past_key_values, add 1 column 1
            batch_input_encode_x_y['attention_mask'] = torch.cat((torch.ones(
                (batch_input_encode_x_y['attention_mask'].shape[0], 1)), batch_input_encode_x_y['attention_mask']),
                                                                 dim=1)
            batch_input_encode_x_y = {k: v.to(self.device) for k, v in batch_input_encode_x_y.items()}
            for i in range(len(sim_score)):
                try:
                    sim_score[i] = torch.cat((sim_score[i], torch.zeros(
                        (batch_input_encode_x_y['labels'][0].shape[0] - sim_score[i].shape[0])).cuda()), dim=0)
                except:
                    print('i', i)
                    print('x_y', x_y)
                    print('x', x)
                    print('y', y)

            sim_score = torch.stack(sim_score)

            num_head = self.model.transformer.config.num_attention_heads
            num_hidden_layers = self.model.transformer.config.num_hidden_layers
            num_dim = self.model.transformer.config.hidden_size
            z = z.repeat(1, 2 * self.model.transformer.config.num_hidden_layers)

            z_past_key_values = z.view(
                z.shape[0],
                1,
                num_hidden_layers * 2,
                num_head,
                num_dim // num_head
            )

            z_past_key_values = z_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

            outputs = self.model(input_ids=batch_input_encode_x_y['input_ids'],
                                 past_key_values=z_past_key_values,
                                 attention_mask=batch_input_encode_x_y['attention_mask'],
                                 labels=batch_input_encode_x_y['labels'], return_dict=True, sim_score=sim_score)

            return outputs['loss']
        else:
            # for i in range(len(x)):
            #     temp = self.tokenizer.eos_token + ' '+ x[i]
            #     x[i] = temp

            batch_input_encode_x = self.tok.batch_encode_plus(
                x,
                max_length=self.max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            batch_input_encode_x = {k: v.to(self.device) for k, v in batch_input_encode_x.items()}
            im = self.model.transformer.wte(batch_input_encode_x['input_ids'])
            e_input = im  # + z.view(-1,1,im.shape[-1])

            output_sequences = []

            input_ids = batch_input_encode_x['input_ids']

            response = []  # According to the context, to generate the response

            # gold_response
            batch_input_encode_y = self.tok.batch_encode_plus(
                y,
                max_length=self.max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            batch_input_encode_p = self.tok.batch_encode_plus(
                p,
                max_length=self.max_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            persona_sequences = []
            context_sequences = []
            gold_sequences = []
            for i in range(batch_input_encode_y['input_ids'].shape[0]):
                persona_sequence = self.tok.convert_ids_to_tokens(batch_input_encode_p['input_ids'][i],
                                                                  skip_special_tokens=True)
                persona_sequences.append(persona_sequence)
                context_sequence = self.tok.convert_ids_to_tokens(batch_input_encode_x['input_ids'][i],
                                                                  skip_special_tokens=True)
                context_sequences.append(context_sequence)
                gold_sequence = self.tok.convert_ids_to_tokens(batch_input_encode_y['input_ids'][i],
                                                               skip_special_tokens=True)
                gold_sequences.append(gold_sequence)

            num_head = self.model.transformer.config.num_attention_heads
            num_hidden_layers = self.model.transformer.config.num_hidden_layers
            num_dim = self.model.transformer.config.hidden_size

            for ba in range(e_input.shape[0]):
                pred_list = []
                response1 =[]
                for i in range(5):
                    # 1,T,D
                    e_input_ba = e_input[ba].unsqueeze(0)
                    z_ba = z[ba].unsqueeze(0)
                    ba_input_ids = input_ids[ba].unsqueeze(0)
                    # 1,T
                    ba_mask = batch_input_encode_x['attention_mask'][ba].unsqueeze(0)
                    ba_mask_sum = torch.sum(ba_mask, dim=1)
                    ba_input_ids = ba_input_ids[:, :ba_mask_sum]
                    ba_mask = ba_mask[:, :ba_mask_sum]
                    e_input_ba = e_input_ba[:, :ba_mask_sum, :]

                    z_ba = z_ba.repeat(1, 2 * self.model.transformer.config.num_hidden_layers)

                    z_ba_past_key_values = z_ba.view(
                        z_ba.shape[0],
                        1,
                        num_hidden_layers * 2,
                        num_head,
                        num_dim // num_head
                    )
                    z_ba_past_key_values = z_ba_past_key_values.permute([2, 0, 3, 1, 4]).split(2)

                    for _ in range(self.config.generate_max_len):
                        outputs = self.model(input_ids=ba_input_ids, past_key_values=z_ba_past_key_values)
                        logits = outputs.logits
                        next_token_logits = logits[0, -1, :]

                        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.config.topk,
                                                                top_p=self.config.topp)
                        a = F.softmax(filtered_logits, dim=-1)
                        for id in set(response1):
                            a[id] /= self.config.repetition_penalty
                        next_token = torch.multinomial(a, num_samples=1)


                        if next_token == self.tok.eos_token_id:  # Meet end indicates the response generated over
                            break

                        response.append(next_token.item())
                        response1.append(next_token.item())

                        ba_input_ids = torch.cat((ba_input_ids, next_token.unsqueeze(0)), dim=1)
                        ba_mask = torch.cat((ba_mask, torch.ones_like(next_token).unsqueeze(0)), dim=1)


                    text = self.tok.convert_ids_to_tokens(response)
                    response = []
                    pred_list.append(text)

                # Use the '###' connect five generate results
                pred_list_all = []
                for pred in pred_list:
                    pred_list_all += pred
                    pred_list_all.append('###')
                pred_list_all = pred_list_all[:-1]
                output_sequences.append(pred_list_all)

            return persona_sequences, context_sequences, output_sequences, gold_sequences