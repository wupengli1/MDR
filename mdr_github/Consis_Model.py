import os


import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import random
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from config import Config

data_language = 'EN'


class Consis_Data(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset_train = None
        self.dataset_valid = None
        self.batch_size = 16
        config = Config()
        # data_language = config.data_language

        self.data_path = config.data_path

        try:
            if data_language=='EN':
                self.tok = BertTokenizer.from_pretrained("F:/premodel/bert-base-uncased",do_lower_case=False)
            else:
                self.tok = BertTokenizer.from_pretrained('F:/premodel/bert-base-chinese')
        except:
            if data_language=='EN':
                self.tok = BertTokenizer.from_pretrained("/media/wupengli/premodel/premodel/bert-base-uncased",do_lower_case=False)
            else:
                self.tok = BertTokenizer.from_pretrained('/media/wupengli/premodel/premodel/bert-base-chinese')
        print(self.tok.sep_token,self.tok.sep_token_id)
        self.tok.sep_token = None
        
        
        self.tok.add_tokens(['[PERSONA]','[CONTEXT]','[RESPONSE]'],special_tokens=True)


    def setup(self,stage):
        
          
        if data_language=='EN':
        # english_data
            with open(os.getcwd() + self.data_path[0] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
        else:
            with open(os.getcwd() + self.data_path[0] + "_train.txt",encoding='utf-8') as f:
                data_single = f.read()
        data_single = data_single.split('\n[SEP]\n')
        data = []

        for i in range(len(data_single)):
            if i+2 < len(data_single):
                temp = data_single[i].split('\n')
                n_temp = data_single[i+1].split('\n')
                n_n_temp = data_single[i+2].split('\n')
                
                temp[1] = temp[1].split('<|endoftext|>')[-2]
                n_temp[1] = n_temp[1].split('<|endoftext|>')[-2]
                n_n_temp[1] = n_n_temp[1].split('<|endoftext|>')[-2]
          

               
                data.append([temp[0],temp[1],temp[2],0])
                
                if temp[0] != n_temp[0]:
                    data.append([temp[0],temp[1],n_temp[2],1])
                
      
                if temp[0] == n_n_temp[0]:
                    data.append([temp[0],temp[1],n_n_temp[2],2])

                
        data_single = data
        
        self.data_single = data_single
        random.shuffle(self.data_single)
        if stage == 'fit' or stage is None:
            self.dataset_train = self.data_single[0:int(len(self.data_single)*0.9)]
            print(len(self.data_single)/16)
            self.dataset_valid = self.data_single[int(len(self.data_single)*0.9):]
        if stage == 'test' or stage is None:
            self.dataset_valid = self.data_single[int(len(self.data_single)*0.9):]



    def collate_fn(self,batch):
        inputs_x = []
        inputs_y = []
        inputs_p = []
        inputs_c = []
        inputs_r = []
        for line in batch:
            inputs_x.append(('[CLS] ' + line[0] + ' [PERSONA] ' + line[1] + ' [CONTEXT] ' + line[2].replace('<|endoftext|>',' [RESPONSE]')))
            inputs_y.append(line[3])
            
            inputs_p.append('[CLS] ' + line[0] + ' [PERSONA]')
            inputs_c.append(line[1] + ' [CONTEXT]')
            inputs_r.append(line[2].replace('<|endoftext|>',' [RESPONSE]'))
        
        
        inputs_x = self.tok.batch_encode_plus(
            inputs_x,
            max_length=256,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
            )
        l_num = 0
        for i in range(len(inputs_x['input_ids'])):
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[PERSONA]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 0
            # inputs_x['input_ids'][i,l_num] = 250
            l_num = temp_l
            
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[CONTEXT]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 1
            l_num = temp_l
            
            temp_l = self.tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False).index('[RESPONSE]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 2

        # print(inputs_x['token_type_ids'][0])     

        inputs = {'input_ids':inputs_x['input_ids'],'attention_mask':inputs_x['attention_mask'],'token_type_ids':inputs_x['token_type_ids']}
        labels = torch.tensor(inputs_y)
        
        return inputs,labels
        
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,num_workers=0,collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,num_workers=0,collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size,num_workers=0,collate_fn=self.collate_fn)

    

class Class_Model(nn.Module):
    def __init__(self) -> None:
        super(Class_Model,self).__init__()
        data_set = Consis_Data()

        self.sentence_embedding = nn.Embedding(3,768)
        self.tok = data_set.tok
        try:
            if data_language=='EN':
                self.model = BertForSequenceClassification.from_pretrained("F:/premodel/bert-base-uncased",num_labels=3)
            else:
                self.model = BertForSequenceClassification.from_pretrained('F:/premodel/bert-base-chinese',num_labels=3)
        except:
            if data_language=='EN':
                self.model = BertForSequenceClassification.from_pretrained("/media/wupengli/premodel/premodel/bert-base-uncased",num_labels=3)
            else:
                self.model = BertForSequenceClassification.from_pretrained('/media/wupengli/premodel/bert-base-chinese',num_labels=3)
        
        self.sentence_embedding.weight.data[1:,:].copy_(self.model.bert.embeddings.token_type_embeddings.weight.data)
        # embedding = self.model.get_input_embeddings()
        # shapesize = embedding.weight.data.shape
        # new_embedding = torch.nn.Embedding(len(self.tok), shapesize[1])
        # new_embedding.weight.data[:shapesize[0], :].copy_(embedding.weight.data)
        self.model.resize_token_embeddings(len(self.tok))

    
    def forward(self,input_ids,attention_mask,token_type_ids,labels=None,return_dict=True):
        
        sentence_embeds = self.sentence_embedding(token_type_ids)
        inputs_embeds = self.model.bert.embeddings.word_embeddings(input_ids)
        inputs_embeds = inputs_embeds + sentence_embeds
        
        
        outputs = self.model(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels = labels,
            return_dict=True)
        
        return outputs


class Consis_Model(pl.LightningModule):
    def __init__(self):
        super(Consis_Model,self).__init__()
        #
        # try:
        #     if data_language == 'EN':
        #         self.tok = BertTokenizer.from_pretrained("F:/premodel/bert-base-uncased", do_lower_case=False)
        #     else:
        #         self.tok = BertTokenizer.from_pretrained('F:/premodel/bert-base-chinese')
        # except:
        #     if data_language == 'EN':
        #         self.tok = BertTokenizer.from_pretrained("/media/wupengli/premodel/premodel/bert-base-uncased",
        #                                                  do_lower_case=False)
        #     else:
        #         self.tok = BertTokenizer.from_pretrained('/media/wupengli/premodel/premodel/bert-base-chinese')

        self.model = Class_Model()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,inputs,labels=None):
        outputs = self.model(input_ids=inputs['input_ids'],
            attention_mask= inputs['attention_mask'],
            token_type_ids = inputs['token_type_ids'],
            labels = labels,
            return_dict=True)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x,y)
        loss = outputs.loss
        self.log('train_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x,y)
        loss = outputs.loss

        y_hat = outputs.logits
        return [y,y_hat,loss]

    def validation_epoch_end(self, outputs):
        for i in range(len(outputs)):
            if i == 0:
                y = outputs[i][0]
                y_hat = outputs[i][1]
                val_loss_sum = outputs[i][2]
            else:
                y = torch.cat([y,outputs[i][0]],dim=0)
                y_hat = torch.cat([y_hat,outputs[i][1]],dim=0)
                val_loss_sum += outputs[i][2]
        val_loss = val_loss_sum/len(outputs)
        self.log('val_loss',val_loss,on_epoch=True,prog_bar=True,logger=True)
        
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat == y).float() / y.shape[0]     
        self.log('val_acc',acc,on_epoch=True,prog_bar=True,logger=True)   

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).logits
        
        return [y,y_hat]

        
    def test_epoch_end(self, outputs):

        for i in range(len(outputs)):
            if i == 0:
                y = outputs[i][0]
                y_hat = outputs[i][1]
            else:
                y = torch.cat([y,outputs[i][0]],dim=0)
                y_hat = torch.cat([y_hat,outputs[i][1]],dim=0)
                
 
        y_hat = torch.argmax(y_hat,dim=1)
        acc = torch.sum(y_hat == y).float() / y.shape[0]
        
        self.log('test_acc',acc,on_epoch=True,prog_bar=True,logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),lr=1e-5,weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=7322, num_training_steps=73220)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        
        return [optimizer], [scheduler]
    
    def predict_dataloader(self):
        return super().predict_dataloader()

def collate_fn_predict(batch):
    try:
        if data_language == 'EN':
            tok = BertTokenizer.from_pretrained("F:/premodel/bert-base-uncased", do_lower_case=False)
        else:
            tok = BertTokenizer.from_pretrained('F:/premodel/bert-base-chinese')
    except:
        if data_language == 'EN':
            tok = BertTokenizer.from_pretrained("/media/wupengli/premodel/premodel/bert-base-uncased",
                                                     do_lower_case=False)
        else:
            tok = BertTokenizer.from_pretrained('/media/wupengli/premodel/premodel/bert-base-chinese')

    tok.sep_token = None

    tok.add_tokens(['[PERSONA]','[CONTEXT]','[RESPONSE]'],special_tokens=True)
    inputs_x = []

    for line in batch:
        inputs_x.append(('[CLS] ' + line[0] + ' [PERSONA] ' + line[1] + ' [CONTEXT] ' + line[2] +' [RESPONSE]'))
        
    inputs_x = tok.batch_encode_plus(
        inputs_x,
        max_length=256,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False,
        )
    
    l_num = 0
    for i in range(len(inputs_x['input_ids'])):
        # print(inputs_x['input_ids'][i])
        temp_str = tok.convert_ids_to_tokens(inputs_x['input_ids'][i],skip_special_tokens=False)
        
        if '[PERSONA]' in temp_str:
            temp_l = temp_str.index('[PERSONA]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 0
            l_num = temp_l
        else:
            inputs_x['token_type_ids'][i,l_num:] = 0
        
        if '[CONTEXT]' in temp_str:
            temp_l = temp_str.index('[CONTEXT]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 1
            l_num = temp_l
        else:
            inputs_x['token_type_ids'][i,l_num:] = 1
        
        if '[RESPONSE]' in temp_str:
            temp_l = temp_str.index('[RESPONSE]')+1
            inputs_x['token_type_ids'][i,l_num:temp_l] = 2
            l_num = temp_l
        else:
            inputs_x['token_type_ids'][i,l_num:] = 2
        

    # print(inputs_x['token_type_ids'][0])     

    inputs = {'input_ids':inputs_x['input_ids'],'attention_mask':inputs_x['attention_mask'],'token_type_ids':inputs_x['token_type_ids']}
    
    labels = None

    
    return inputs,labels    
    
def predict(batch,lang='en',model_path_en="../1-en-1/models/consis_model_EN.ckpt",model_path_zh="../1-en-1/models/consis_model_ZH.ckpt"):

    predict_dataloader = DataLoader(batch, batch_size=1,num_workers=0,collate_fn=collate_fn_predict)

    model = Consis_Model()
    model = model.to(torch.device("cuda:0"))
    if lang=='en':
        d = torch.load(model_path_en)
    else:
        d = torch.load(model_path_zh)
    model.load_state_dict(d['state_dict'],strict=False)
    model.eval()
    
    y_predict = []
    for i, batch in enumerate(predict_dataloader):
        x, y = batch
        x = {k: v.to(torch.device("cuda:0")) for k, v in x.items()}
        y_hat = model(x).logits
        y_hat = torch.argmax(y_hat,dim=1)
        y_hat = y_hat.tolist()
        y_predict.extend(y_hat)
    
    return y_predict
        
    

if __name__=='__main__':

    model_file = 'T-Consis_Model_ZH'
    dm = Consis_Data()
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only = True,
        filename= model_file + '-{epoch:02d}-{step}-{val_loss:.2f}-{val_acc:.2f}',
        save_last=True
    )
    learning_rate_callback = LearningRateMonitor()
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.0, 
        patience=3, 
        mode='min', 
        strict=True
    )
    logger = TensorBoardLogger('log_consis', name='test')
    
    dm.setup('fit')
    trainer = pl.Trainer(
        default_root_dir='./models',
        gradient_clip_val=0.5,
        max_epochs=10,
        gpus=1,
        val_check_interval=0.5,
        callbacks=[learning_rate_callback, checkpoint_callback,early_stopping],
        logger = logger,
    )  
    model = Consis_Model()
    trainer.fit(model, datamodule=dm)
    
    dm.setup('test')
    
    trainer.test(model, datamodule=dm)

