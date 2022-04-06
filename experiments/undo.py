import argparse
import os

import torch
from experiments.experiment import Experiment
import data
from torch.utils.data import DataLoader
from models import Simple
from torch import nn
from torch import optim

class UndoExperiment(Experiment):
    @staticmethod
    def add_parser(parser:argparse.ArgumentParser):
        parser.add_argument("--root",type=str,help="root location for the experiment",required=True)
        parser.add_argument("--dataset",type=str,help="path of the dataset",required=True)
        subparser = parser.add_subparsers(dest="mode")
        train_parser = subparser.add_parser("train")
        train_parser.add_argument("--epochs",type=int,default=50)
        train_parser.add_argument("--batch_size",type=int,default=16)
        return parser

    def __init__(self, args:argparse.Namespace) -> None:
        self.root = args.root
        self.dataset = data.UndoData(args.dataset)
        self.mode = args.mode
        self.model = Simple(2622)
        if self.mode=="train":
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(0)
        

    def run(self):
        if self.mode == "train":
            self.train()

    def train(self):
        print(self.device)
        self.model.to(device=self.device)
        train_set, val_set = torch.utils.data.random_split(self.dataset,[800,200])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,drop_last=True)
        
        for epoch in range(self.epochs):
            l = self.run_epoch(dataloader=train_loader)
            v = self.validate(dataloader=val_loader)
            print(f"Train:{l} Val:{v}")

    def run_epoch(self,dataloader:DataLoader):
        steps = 0
        avg_loss = 0
        for original, altered, undo in dataloader:
            self.optimizer.zero_grad()
            original=original.float().to(self.device)
            altered=altered.float().to(self.device)
            undo=undo.float().to(self.device)

            pos_output = self.model(original)
            pos_loss = self.criterion(pos_output,torch.ones([self.batch_size]).type(torch.LongTensor).to(self.device))
            
            neg_output = self.model(altered)
            neg_loss = self.criterion(neg_output,torch.zeros([self.batch_size]).type(torch.LongTensor).to(self.device))

            undo_output = self.model(undo)
            undo_loss = self.criterion(undo_output,torch.ones([self.batch_size]).type(torch.LongTensor).to(self.device))

            loss = pos_loss+2*neg_loss+undo_loss
            loss.backward()
            self.optimizer.step()
            avg_loss+=loss.item()
            steps+=self.batch_size
        return avg_loss/steps

    def validate(self, dataloader:DataLoader):
        steps = 0
        avg_loss = 0
        c_matrix = torch.tensor([[0,0],[0,0]])
        with torch.no_grad():
            for original, altered, undo in dataloader:
                original=original.float().to(self.device)
                altered=altered.float().to(self.device)
                undo=undo.float().to(self.device)


                pos_output = self.model(undo)
                tp = (pos_output[:,1]>pos_output[:,0]).sum().item()
                fn = (pos_output[:,1]<=pos_output[:,0]).sum().item()
                
                pos_loss = self.criterion(pos_output,torch.ones([self.batch_size]).type(torch.LongTensor).to(self.device))
                
                neg_output = self.model(altered)
                tn = (neg_output[:,1]<=neg_output[:,0]).sum().item()
                fp = (neg_output[:,1]>neg_output[:,0]).sum().item()


                neg_loss = self.criterion(neg_output,torch.zeros([self.batch_size]).type(torch.LongTensor).to(self.device))

                c_matrix+=torch.tensor([[tn,fp],[fn,tp]])
                
                loss = pos_loss+neg_loss
                avg_loss+=loss.item()
                steps+=self.batch_size
        print(c_matrix)
        return avg_loss/steps
            

    def test(self):
        pass

    