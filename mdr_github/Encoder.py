from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from config import Config
import pytorch_lightning as pl
from modeling_gpt2_1 import GPT2LMHeadModel


class Encoder(pl.LightningModule):
    r""" The encoder """

    def __init__(self):
        super(Encoder, self).__init__()
        config = Config()
        data_language = config.data_language
        if data_language == 'EN':
            self.tok = GPT2Tokenizer.from_pretrained("F:/premodel/gpt2", do_lower_case=False)
            self.model = GPT2LMHeadModel.from_pretrained("F:/premodel/gpt2")
            self.tok.pad_token = '<pad>'
        else:
            self.tok = BertTokenizer.from_pretrained(
                "F:/premodel/gpt2chinesecluecorpussmall")  # uer/gpt2-chinese-cluecorpussmall
            self.model = GPT2LMHeadModel.from_pretrained("F:/premodel/gpt2chinesecluecorpussmall",
                                                         ignore_mismatched_sizes=True)
            self.tok.eos_token = '[SEP]'

        self.config = Config()
        self.max_length = self.config.max_len

    def forward(self, inputs):

        batch_input_encode = self.tok.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        # cuda
        batch_input_encode = {k: v.to(self.device) for k, v in batch_input_encode.items()}

        outputs = self.model(input_ids=batch_input_encode['input_ids'],
                             attention_mask=batch_input_encode['attention_mask'],
                             output_hidden_states=True,
                             return_dict=True,
                             )

        pooler_output = outputs.hidden_states[-1][:, -1, :]  # [batch, dim]

        return pooler_output
