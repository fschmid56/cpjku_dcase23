import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import torch.nn as nn

from datasets.dcase23 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from models.cp_mobile_clean import get_model
from models.mel import AugmentMelSTFT
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.utils import mixstyle
from helpers import nessi


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse and contains all configurations for our experiment
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        # CP-Mobile - our model to be trained on the log mel spectrograms
        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )

        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

    def mel_forward(self, x):
        """
        @param x: a batch of raw signals (waveform)
        return: a batch of log mel spectrograms
        """
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])  # for calculating log mel spectrograms we remove the channel dimension
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])  # batch x channels x mels x time-frames
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """
        x, file, labels, devices, cities, teacher_logits = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)

        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        label_loss = samples_loss.mean()

        # Temperature adjusted probabilities of teacher and student
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.config.temperature, dim=-1)

        kd_loss = self.kl_div_loss(y_hat_soft, teacher_logits).mean()
        kd_loss = kd_loss * (self.config.temperature ** 2)
        loss = self.config.kd_lambda * label_loss + (1 - self.config.kd_lambda) * kd_loss

        results = {"loss": loss, "label_loss": label_loss * self.config.kd_lambda,
                   "kd_loss": kd_loss * (1 - self.config.kd_lambda)}

        return results

    def training_epoch_end(self, outputs):
        """
        :param outputs: contains the items you log in 'training_step'
        :return: a dict containing the metrics you want to log to Weights and Biases
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_label_loss = torch.stack([x['label_loss'] for x in outputs]).mean()
        avg_kd_loss = torch.stack([x['kd_loss'] for x in outputs]).mean()
        self.log_dict({'loss': avg_loss, 'label_loss': avg_label_loss, 'kd_loss': avg_kd_loss})

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        x = self.mel_forward(x)
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'val_loss': loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_pred_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

        logs = {'val_acc': val_acc, 'val_loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            dev_corrct = torch.stack([x["devn_correct." + d] for x in outputs]).sum()
            logs["vloss." + d] = dev_loss / dev_cnt
            logs["vacc." + d] = dev_corrct / dev_cnt
            logs["vcnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = torch.stack([x["lblloss." + l] for x in outputs]).sum()
            lbl_cnt = torch.stack([x["lblcnt." + l] for x in outputs]).sum()
            lbl_corrct = torch.stack([x["lbln_correct." + l] for x in outputs]).sum()
            logs["vloss." + l] = lbl_loss / lbl_cnt
            logs["vacc." + l] = lbl_corrct / lbl_cnt
            logs["vcnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["vacc." + l] for l in self.label_ids]))
        self.log_dict(logs)


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="CPJKU pipeline for DCASE23 Task 1.",
        tags=["DCASE23"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    # train dataloader
    train_dl = DataLoader(dataset=get_training_set(config.cache_path, config.resample_rate, config.roll,
                                                   config.dir_prob, config.temperature),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    # ATTENTION: this is before layer fusion, therefore the MACs and Params slightly deviate from what is
    # reported in the challenge submission
    sample = next(iter(train_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_model_size(pl_module.model, input_size=shape)
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=1,
                         callbacks=[lr_monitor])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE23_Task1")
    parser.add_argument('--experiment_name', type=str, default="CPJKU_Training")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders

    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default="datasets/cpath")

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale CP-Mobile)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=int, default=2.3)
    parser.add_argument('--expansion_rate', type=int, default=3)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--roll', type=int, default=4000)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.6)  # prob. to apply device impulse response augmentation
    ## knowledge distillation
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--kd_lambda', type=float, default=0.02)

    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=0.0009)
    parser.add_argument('--warm_up_len', type=int, default=14)
    parser.add_argument('--ramp_down_start', type=int, default=50)
    parser.add_argument('--ramp_down_len', type=int, default=84)
    parser.add_argument('--last_lr_value', type=float, default=0.005)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_size', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()
    train(args)
