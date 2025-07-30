import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
print(sys.path)

import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import pytorch_lightning as pl
import random
import warnings
from model_1 import MLModel
from config import Config
from Matrics import NLP_Gen_Matrics
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")
resnumber='model1'

class Net(pl.LightningModule):
    def __init__(
            self,
            batch_size,
            epochs,
            t_total=100000,
            data_path=[],
            max_length=512,
            warm_up_steps=0,
            lr=1e-4,
            check_test=False,
            model_file='None',
    ):
        super(Net, self).__init__()
        self.check_test = check_test

        self.batch_size = batch_size
        self.epochs = epochs
        # The total number
        self.t_total = t_total
        # Preheat steps
        self.warm_up_steps = warm_up_steps
        self.lr = lr
        self.config = Config()
        self.config.model_file = model_file
        self.model_file = model_file
        self.model = MLModel(self.config)
        self.max_length = max_length
        self.data_path = data_path
        self.real_data_num = 0
        self.g_step = 0
        self.data_language = self.config.data_language
        self.model.to(self.device)

    def forward(self, inputs, is_train):
        outputs = self.model(inputs=inputs, is_train=is_train)
        return outputs

    # Data processing in advance
    def prepare_data(self):
        # The current working directory
        print(os.getcwd())
        if self.data_language == 'EN':
            # english_data
            with open(os.getcwd() + self.data_path[0] + "_train.txt", encoding='utf-8') as f:
                data_single = f.read()
            data_single_train = data_single.split('\n[SEP]\n')
            with open(os.getcwd() + self.data_path[0] + "_test.txt", encoding='utf-8') as f:
                data_single = f.read()
            data_single_test = data_single.split('\n[SEP]\n')
            with open(os.getcwd() + self.data_path[0] + "_val.txt", encoding='utf-8') as f:
                data_single = f.read()
            data_single_val = data_single.split('\n[SEP]\n')
            eos_token = '<|endoftext|>'
        else:

            with open(os.getcwd() + self.data_path[1] + "_train.txt", encoding='utf-8') as f:
                data_single_train = f.read()
            data_single_train = data_single_train.replace('[SEP]', '[SPE]')
            data_single_train = data_single_train.replace('<|endoftext|>', '[SEP]')
            data_single_train = data_single_train.split('\n[SPE]\n')

            with open(os.getcwd() + self.data_path[1] + "_test.txt", encoding='utf-8') as f:
                data_single_test = f.read()
            data_single_test = data_single_test.replace('[SEP]', '[SPE]')
            data_single_test = data_single_test.replace('<|endoftext|>', '[SEP]')
            data_single_test = data_single_test.split('\n[SPE]\n')
            eos_token = '[SEP]'

        data_train = []
        len_data_single_train = len(data_single_train)
        for line_i in range(len_data_single_train):
            line = data_single_train[line_i]
            temp = line.split('\n')
            temp1_split = temp[1].split(eos_token)
            history = [i for i in temp1_split[1:-2]]
            history_t = []
            if len(history) % 2 == 0:
                for his_i in range(0, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            else:
                history_t.append(history[0])
                for his_i in range(1, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            temp[1] = temp1_split[-2].strip()
            if temp[1] == '':
                continue
            if temp[1][-1] not in '!.?)"\'':
                temp[1] += '.'
            temp[1] = temp[1] + eos_token
            temp[2] = temp[2].split(eos_token)[0].strip()
            if temp[2][-1] not in '!.?)"\'':
                temp[2] += '.'
            temp[2] = temp[2] + eos_token
            data_train.append(temp + [temp[1] + temp[2]] + [history_t])
        data_single_train = data_train

        data_test = []
        for line in data_single_test:
            temp = line.split('\n')
            temp1_split = temp[1].split(eos_token)
            history = [i for i in temp1_split[1:-2]]
            history_t = []
            if len(history) % 2 == 0:
                for his_i in range(0, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            else:
                history_t.append(history[0])
                for his_i in range(1, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            temp[1] = temp1_split[-2].strip()
            if temp[1] == '':
                continue
            if temp[1][-1] not in '!.?)"\'':
                temp[1] += '.'
            temp[1] = temp[1] + eos_token
            temp[2] = temp[2].split(eos_token)[0].strip()
            if temp[2][-1] not in '!.?)"\'':
                temp[2] += '.'
            temp[2] = temp[2] + eos_token

            data_test.append(temp + [temp[1] + temp[2]] + [history_t])
        data_single_test = data_test
        data_val = []
        for line in data_single_val:
            temp = line.split('\n')
            temp1_split = temp[1].split(eos_token)
            history = [i for i in temp1_split[1:-2]]
            history_t = []
            if len(history) % 2 == 0:
                for his_i in range(0, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            else:
                history_t.append(history[0])
                for his_i in range(1, len(history) - 1, 2):
                    history_t.append(history[his_i] + history[his_i + 1])
            temp[1] = temp1_split[-2].strip()
            if temp[1] == '':
                continue
            if temp[1][-1] not in '!.?)"\'':
                temp[1] += '.'
            temp[1] = temp[1] + eos_token
            temp[2] = temp[2].split(eos_token)[0].strip()
            if temp[2][-1] not in '!.?)"\'':
                temp[2] += '.'
            temp[2] = temp[2] + eos_token
            data_val.append(temp + [temp[1] + temp[2]] + [history_t])
        data_single_val = data_val
        if self.check_test:
            data_single_train = data_single_train[:1000]
            data_single_test = data_single_test[:30]
        random.shuffle(data_single_train)
        self.dataset_train = data_single_train  # [:32]
        self.dataset_valid = data_single_val  # [:32]
        self.dataset_test = data_single_test#[:32]
        self.small_test = self.dataset_valid[-10:]

        self.real_data_num = len(self.dataset_train)
        self.t_total = self.epochs * self.real_data_num // self.batch_size
        self.warm_up_steps = self.t_total // 5
        print('Data sample:', self.dataset_train[0])
        print('train:', len(self.dataset_train))
        print('val:', len(self.dataset_valid))
        print('test:', len(self.dataset_test))

    def collate_fn_text(self, batch):
        # batch Internal triples persona,x,y
        inputs = {'persons': [], 'posts': [], 'responses': [], 'p_r': [], 'his': []}
        for line in batch:
            inputs['persons'].append(line[0])
            inputs['posts'].append(line[1])
            inputs['responses'].append(line[2])
            inputs['p_r'].append(line[3])
            inputs['his'].append(line[4])
        return inputs

    # The data load
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn_text,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn_text,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_fn_text,
        )

    # The optimizer
    def configure_optimizers(self):
        print('学习率预热:', self.warm_up_steps, self.t_total)
        optimizer_g = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)
        scheduler_g = get_linear_schedule_with_warmup(
            optimizer_g, self.warm_up_steps, self.t_total
        )
        scheduler_g = {"scheduler": scheduler_g, "interval": "step", "frequency": 1}
        return [optimizer_g], [scheduler_g]

    def training_step(self, batch, batch_nb):
        tensorboard = self.logger.experiment
        loss = self.forward(batch, is_train=True)
        loss_dict = {'train_loss': loss}
        for key in loss_dict:
            self.log(key, loss_dict[key], on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=batch_nb)
        tensorboard.add_scalars('loss_dict', loss_dict, self.global_step)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.forward(batch, is_train=True)
        # loss, decode_loss, kld_loss, di_loss = self.compute_loss(outputs)
        loss_dict = {'val_loss': loss}
        return loss_dict
    def validation_epoch_end(self, outputs):
        # On average, the output is a list of each element in the list of validation_step return values
        val_loss_list = []
        for i in range(len(outputs)):
            val_loss_list.append(outputs[i]['val_loss'])
        # averaging
        avg_val_loss = torch.stack(val_loss_list).mean()
        self.log(
            "avg_val_loss",
            avg_val_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_nb):
        persona_sequences, context_sequences, output_sequences, gold_sequences = self.forward(batch, is_train=False)
        return {'persona': persona_sequences, 'context': context_sequences, 'pred': output_sequences,
                'target': gold_sequences}
    def test_epoch_end(self, outputs) -> None:
        persona_sequences = []
        context_sequences = []
        output_sequences = []
        gold_sequences = []
        for i in range(len(outputs)):
            persona_sequences += outputs[i]['persona']
            context_sequences += outputs[i]['context']
            output_sequences += outputs[i]['pred']
            gold_sequences += outputs[i]['target']
        # save the result
        # format: 'I [CSE] am [CSE] you \ n my [CSE] is [CSE] he \ n \ n'
        pred_list = []
        for i in range(len(output_sequences)):
            pred_list.append(
                '[CSE]'.join(persona_sequences[i]) + '\n' + '[CSE]'.join(context_sequences[i]) + '\n' + '[CSE]'.join(
                    output_sequences[i]) + '\n' + '[CSE]'.join(gold_sequences[i]))

        if not os.path.exists(os.getcwd() + '/result' + resnumber+'/'):
            os.mkdir(os.getcwd() + '/result' + resnumber+'/')
        with open(os.getcwd() + '/result' + resnumber+'/' + self.model_file + '-pred_result.txt', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(pred_list))

if __name__ == "__main__":
    config = Config()
    # seed
    pl.seed_everything(config.seed)
    check_test = False
    model_name = 'MDRGen'
    data_language = config.data_language
    model_file = model_name + '_' + data_language
    print('model_file:', model_file)
    config.model_file = model_file
    max_length = config.max_len
    batch_size = config.batch_size
    accumulate_grad_batches = config.accumulate_grad_batches

    epochs = config.epochs
    output_path = config.output_dir1
    lr = config.lr
    data_path = config.data_path
    logger = TensorBoardLogger('tb_logs', name=model_file)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        save_top_k=1,
        monitor="avg_val_loss",
        mode="min",
        save_weights_only=True,
        filename=model_file + '-{epoch:02d}-{step}-{avg_val_loss:.2f}',
    )
    learning_rate_callback = LearningRateMonitor()

    early_stopping = EarlyStopping(
        monitor='avg_val_loss',
        min_delta=0.0,
        patience=3,
        mode='min',
        strict=True
    )

    net = Net(
        batch_size,
        epochs,
        data_path=data_path,
        max_length=max_length,
        lr=lr,
        check_test=check_test,
        model_file=model_file,
    )
    trainer = pl.Trainer(
        default_root_dir=output_path,
        accumulate_grad_batches = accumulate_grad_batches,
        gradient_clip_val=0.5,
        max_epochs=epochs,
        gpus=1,
        val_check_interval=0.5,
        callbacks=[learning_rate_callback, checkpoint_callback, early_stopping],
        logger=logger,
    )
    # for p in net.model.encoder1.model.parameters():
    #     print(p[0,:10])
    #     break
    best_model_path = './Model/' + os.listdir('./Model')[-1]
    print(best_model_path)
    d = torch.load(best_model_path)["state_dict"]
    net.load_state_dict(d, strict=False)
    trainer.fit(net)

    best_model_path = './Model_1/' + os.listdir('./Model_1')[-1]
    d = torch.load(best_model_path)["state_dict"]
    print('best_model_path:', best_model_path)
    net.load_state_dict(d, strict=False)
    trainer.test(model=net)
    judger = NLP_Gen_Matrics(model_name=model_file, condition_list=['2023-01-01'],resnumber=resnumber)
    result = judger.get_judge_data('./result' + resnumber+'/'+ config.model_file + '-pred_result.txt')
