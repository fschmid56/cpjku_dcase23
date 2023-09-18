import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import os
from torch.hub import download_url_to_file

from datasets.dcase23 import get_test_set
from helpers.init import worker_init_fn
from models.cp_resnet import get_model as get_cpresnet
from models.passt import get_model as get_passt
from models.mel import AugmentMelSTFT


pretrained_models_url = "https://github.com/fschmid56/cpjku_dcase23/releases/download/ensemble_logits/"


class PLModule(pl.LightningModule):
    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.mel = AugmentMelSTFT(**model_config['mel'])

        get_model_fn = model_config['model_fn']
        self.model = get_model_fn(**model_config["net"])

        # load pre-trained model parameters
        state_dict_file = os.path.join("resources", f"{config.model_name}.pt")
        if not os.path.isfile(state_dict_file):
            print("Download pre-trained weights.")
            download_url_to_file(os.path.join(pretrained_models_url, f"{config.model_name}.pt"), state_dict_file)
        pretrained_weights = torch.load(state_dict_file)
        self.model.load_state_dict(pretrained_weights)

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

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch
        x = self.mel_forward(x)
        model_out = self.model(x)
        if len(model_out) == 2:
            y_hat = model_out[0]
        else:
            y_hat = model_out
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


def validate(config, model_config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="CPJKU pipeline for DCASE23 Task 1.",
        tags=["DCASE23"],
        config=config,  # this logs the hyperparameters for us
        name=config.experiment_name
    )

    # test loader
    test_dl = DataLoader(dataset=get_test_set(config.cache_path, model_config['mel']['sr']),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, model_config)

    trainer = pl.Trainer(logger=wandb_logger,
                         accelerator='auto',
                         devices=1
                         )
    # start training and validation for the specified number of epochs
    trainer.validate(pl_module, dataloaders=test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE23_Task1")
    parser.add_argument('--experiment_name', type=str, default="CPJKU_Teacher_Validation")
    parser.add_argument('--num_workers', type=int, default=12)  # number of workers for dataloaders

    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))
    parser.add_argument('--batch_size', type=int, default=32)

    # model
    parser.add_argument('--model_name', type=str, default="passt_dirfms_1")

    args = parser.parse_args()

    if args.model_name in ["cpr_128k_dirfms_1",
                           "cpr_128k_dirfms_2",
                           "cpr_128k_dirfms_3",
                           "cpr_128k_fms_1",
                           "cpr_128k_fms_2",
                           "cpr_128k_fms_3"]:
        model_config = {
            "mel": {
                "sr": 32000,
                "n_mels": 256,
                "win_length": 3072,
                "hopsize": 750,
                "n_fft": 4096,
                "fmax": None,
                "fmax_aug_range": 1000,
                "fmin": 0,
                "fmin_aug_range": 1
            },
            "net": {
                # "rho": 8,
                # "base_channels": 32,
                # "maxpool_stage1": [1],
                # "maxpool_kernel": (2, 1),
                # "maxpool_stride": (2, 1)
            },
            "model_fn": get_cpresnet
        }
    elif args.model_name in ["passt_dirfms_1",
                             "passt_dirfms_2",
                             "passt_dirfms_3",
                             "passt_fms_1",
                             "passt_fms_2",
                             "passt_fms_3"]:
        model_config = {
            "mel": {
                "sr": 32000,
                "n_mels": 128,
                "win_length": 800,
                "hopsize": 320,
                "n_fft": 1024,
                "fmax": None,
                "fmax_aug_range": 1000,
                "fmin": 0,
                "fmin_aug_range": 1
            },
            "net": {
                "arch": "passt_s_swa_p16_128_ap476",
                "n_classes": 10,
                "input_fdim": 128,
                "s_patchout_t": 0,
                "s_patchout_f": 6
            },
            "model_fn": get_passt
        }
    else:
        raise NotImplementedError(f"No model with model name {args.model_name} in resources folder!")

    validate(args, model_config)
