import argparse
from math import inf
from experiments.experiment import Experiment
import torch
from torch import nn, optim, Tensor
import data
import models
from tqdm import tqdm
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import os
import utils
import matplotlib.pyplot as plt
import numpy as np


class MobileNetAlteration(Experiment):
    @staticmethod
    def add_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--root", type=str, help="root location for the experiment", required=True)
        parser.add_argument("--dataset", type=str,
                            help="path of the dataset", required=True)
        parser.add_argument("--subset", nargs="*",
                            help="which subset of alterations to be used")
        subparser = parser.add_subparsers(dest="mode", required=True)

        train_parser = subparser.add_parser("train")
        train_parser.add_argument("--epochs", type=int, default=20)
        train_parser.add_argument("--batch_size", type=int, default=32)
        train_parser.add_argument("--save", action="store_true")
        train_parser.add_argument("--save_freq", type=int, default=10)
        train_parser.add_argument(
            "--weight", type=int, nargs=3, default=[1, 1, 1])
        train_parser.add_argument("--lr", type=float, default=0.001)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args.root)
        self.dataset = data.AlterationData(args.dataset, args.subset)
        print(f"Size of dataset is {len(self.dataset)}")
        self.mode = args.mode
        self.model = models.ImageAlteration(
            tail_sizes=[500, 13], num_classes=3)
        # print(self.model)
        if self.mode == "train":
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.weight = torch.tensor(args.weight).float()
            self.lr = args.lr
            self.save = 0
            if args.save:
                self.save = args.save_freq
                self.savedir = os.path.join(self.root, "checkpoints")
                os.makedirs(self.savedir)

            print("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            self.criterion = nn.CrossEntropyLoss(weight=self.weight)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            torch.manual_seed(0)

    def run(self):
        if self.mode == "train":
            self.train()

    def train(self):
        print(self.device)
        with SummaryWriter(os.path.join(self.root, "logs")) as logger:
            self.model.to(device=self.device)
            self.criterion = self.criterion.to(device=self.device)
            length = len(self.dataset)
            train_len = int(length*0.8)
            val_len = length-train_len
            train_set, val_set = random_split(
                self.dataset, [train_len, val_len])
            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
            temp_iter = iter(train_loader)
            temp_data = temp_iter.next()
            logger.add_graph(
                self.model, temp_data["im_orig"].float().to(self.device))
            pbar = tqdm(range(self.epochs))
            min_val_loss = inf
            for epoch in pbar:
                l = self.train_epoch(dataloader=train_loader)
                logger.add_scalar("loss/train", l, epoch)

                v, recon_matrix, alter_matrix = self.validate(
                    dataloader=val_loader)
                logger.add_scalar("loss/validation", v, epoch)

                # (print(recon_matrix))
                for i, att in enumerate(self.dataset.get_att_names()):
                    fig = utils.mat_to_figure(recon_matrix[i, :, :], [
                                              "Deleted", "No change", "Added"], ["Deleted", "No change", "Added"])
                    logger.add_figure(
                        "confusion_mtrix/reconstruction/{}".format(att), fig, epoch)
                    plt.close()

                    fig = utils.mat_to_figure(alter_matrix[i, :, :], [
                                              "Deleted", "No change", "Added"], ["Deleted", "No change", "Added"])
                    logger.add_figure(
                        "confusion_mtrix/alteration/{}".format(att), fig, epoch)
                    plt.close()
                    min_val_loss = min(min_val_loss, v)
                # v = 0
                if self.save > 0 and epoch % self.save == 0:
                    torch.save({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "train_loss": l,
                        "val_loss": v,
                        "recon_matrix": recon_matrix,
                        "alter_matrix": alter_matrix
                    }, os.path.join(self.savedir, f"{epoch}.pth"))
                tqdm.write(
                    f"Epoch: {epoch:3d} Train: {l:.16f} Val: {v:.16f} Min_Val: {min_val_loss:.16f}")
            logger.add_hparams(
                {
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "weight": str(self.weight),
                    "batch_size": self.batch_size
                },
                metric_dict={
                    "Min Validation Loss": min_val_loss
                }
            )

    def train_epoch(self, dataloader: DataLoader):
        self.model.train(True)
        steps = 0
        avg_loss = 0
        pbar = tqdm(dataloader, leave=False)
        for sample_batch in pbar:
            self.optimizer.zero_grad()
            reconstruction = sample_batch["im_recon"].float().to(self.device)
            altered = sample_batch["im_alter"].float().to(self.device)
            change = sample_batch["change"].long().to(self.device)

            change = change+1

            orig_change = (torch.ones(change.shape)).long().to(self.device)

            orig_out = self.model(reconstruction)
            orig_loss = self.criterion(orig_out, orig_change)

            alter_out = self.model(altered)
            alter_loss = self.criterion(alter_out, change)

            # # loss = orig_loss
            loss = orig_loss + alter_loss

            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
            steps += 2
            # break
            # # steps+=num_samples
            pbar.set_description(f"Training: Loss:{avg_loss/steps:.16f}")

        return avg_loss/steps

    def validate(self, dataloader: DataLoader):
        steps = 0
        avg_loss = 0
        # c_matrix = torch.tensor([[0,0],[0,0]])
        recon_matrix = np.zeros([len(self.dataset.get_att_names()), 3, 3])
        alter_matrix = np.zeros([len(self.dataset.get_att_names()), 3, 3])
        with torch.no_grad():
            self.model.train(False)
            pbar = tqdm(dataloader, leave=False)
            pbar.set_description("Validation")
            for sample_batch in pbar:
                reconstruction = sample_batch["im_recon"].float().to(
                    self.device)
                altered = sample_batch["im_alter"].float().to(self.device)
                change = sample_batch["change"].long().to(self.device)

                change = change+1

                orig_change = (torch.ones(change.shape)).long().to(self.device)

                orig_out = self.model(reconstruction)
                # tqdm.write(str(orig_out.shape))
                orig_loss = self.criterion(orig_out, orig_change)

                alter_out = self.model(altered)
                alter_loss = self.criterion(alter_out, change)

                for i in range(len(self.dataset.get_att_names())):
                    recon_matrix[i, :, :] += self.get_conf_matrix(
                        nn.functional.softmax(orig_out[:, :, i], dim=1), orig_change[:, i])
                    alter_matrix[i, :, :] += self.get_conf_matrix(
                        nn.functional.softmax(alter_out[:, :, i], dim=1), change[:, i])
                # loss = orig_loss
                # k = 2
                # print()
                # print(orig_out[k,:].sigmoid())
                # print(alter_out[k,:].sigmoid())
                # print(change[k,:])
                loss = orig_loss + alter_loss
                avg_loss += loss.item()

                # because we have calculated loss twice (orig and alter)
                # The base line loss should be ~ln2 = 0.69..
                steps += 2

        # print(c_matrix)
        recon_matrix /= recon_matrix.sum(axis=-1, keepdims=True)
        alter_matrix /= alter_matrix.sum(axis=-1, keepdims=True)
        return avg_loss/steps, recon_matrix, alter_matrix

    @staticmethod
    def get_conf_matrix(x: Tensor, y: Tensor):
        c = x.shape[1]
        cm = np.zeros([c, c])
        # print(x)
        x_t = x.cpu().numpy()
        y_t = y.cpu().numpy()
        for i in range(c):
            cm[i, :] = x_t[y_t == i].sum(axis=0)

        return cm
