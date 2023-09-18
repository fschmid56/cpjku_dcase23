import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import os

from helpers.utils import mixstyle
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.init import worker_init_fn
from models.cp_resnet import get_model
from models.mel import AugmentMelSTFT
from helpers import nessi
from datasets.dcase23 import get_training_set, get_test_set

class PLModule(pl.LightningModule):
    def __init__(self, config):
        super(PLModule, self).__init__()

        self.config = config
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

        self.model = get_model(base_channels=config.base_channels)

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        self.epoch = 0

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, files, labels, devices, cities, teacher_logits = batch

        if self.mel:
            x = self.mel_forward(x)

        # mixstyle
        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # forward
        y_hat = self.forward(x)

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()
        samples_loss = samples_loss.detach()

        devices = [d.rsplit("-", 1)[1][:-4] for d in files]

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == labels).sum()
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            if d[0] == "i":
                if self.epoch == 0 and batch_idx < 10:
                    print(f"device {d} ignored!")
                continue
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devcnt." + d] = results["devcnt." + d] + 1.
        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'train.loss': avg_loss, 'train_acc': train_acc}

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            logs["tloss." + d] = dev_loss / dev_cnt
            logs["tcnt." + d] = dev_cnt

        self.log_dict(logs)

        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {train_acc}")

    def validation_step(self, batch, batch_idx):
        x, files, labels, devices, cities = batch

        if self.mel:
            x = self.mel_forward(x)

        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()

        devices = [d.rsplit("-", 1)[1][:-4] for d in files]

        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc}

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

        self.log_dict(logs)

        if self.epoch > 0:
            print()
            print(f"Validation Loss: {avg_loss}")
            print(f"Validation Accuracy: {val_acc}")

        self.epoch += 1


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
                                                   config.dir_prob),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # test loader
    test_dl = DataLoader(dataset=get_test_set(config.cache_path, config.resample_rate),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # wandb_logger.log({"batch_size": config.batch_size})

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
    parser.add_argument('--experiment_name', type=str, default="CPJKU_cp-resnet_teacher_training")
    parser.add_argument('--num_workers', type=int, default=16)  # number of workers for dataloaders

    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.8)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--roll', type=int, default=4_000)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0.4)  # prob. to apply device impulse response augmentation

    # model
    parser.add_argument('--base_channels', type=int, default=32)  # base channels of model

    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warm_up_len', type=int, default=15)
    parser.add_argument('--ramp_down_start', type=int, default=50)
    parser.add_argument('--ramp_down_len', type=int, default=85)
    parser.add_argument('--last_lr_value', type=float, default=0.005)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=3072)  # in samples
    parser.add_argument('--hop_size', type=int, default=750)  # in samples
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()

    train(args)
