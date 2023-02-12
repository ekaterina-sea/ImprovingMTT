import pandas as pd
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import random

from SI2_dataprep import prepare_dataset, load_dataset

class DNNModel(pl.LightningModule):
    def __init__(self,task='regression',DNNlayers=[512,256,128],dropout=[0.0,0.0,0.0],actF='1',optF='RAdam',lr=0.001,idx2target=None,idx2compid=None,nBits=1024, FileName='FileName',seed=70922,savestep=10):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.FileName = FileName
        self.task = task
        self.lr = lr
        self.predT = np.array(['Pred'])
        self.trueT = np.array(['True'])
        self.targID = np.array(['TargetID'])
        self.compID = np.array(['CompoundID'])
        self.NIter = np.array(['Iter'])
        self.idx2target = idx2target
        self.idx2compid = idx2compid
        self.DNNlayers = DNNlayers
        print('DNNlayers',self.DNNlayers)
        self.dropout = dropout
        print('dropout',self.dropout)
        activation_list=nn.ModuleDict({"0":nn.LeakyReLU(),"1":nn.PReLU(),"2":nn.ELU(),"3":nn.Hardshrink()})
        self.actF = activation_list[str(actF)].to(self.device)
        self.optF = optF
        self.savestep = savestep
        print(self.actF)
        if self.task=='regression':
            modules = []
            l_1st=nBits
            for l,do in zip(self.DNNlayers,self.dropout):
                modules.append(nn.Linear(l_1st, l))
                modules.append(self.actF)
                modules.append(nn.LayerNorm(l))
                modules.append(nn.Dropout(float(do)))
                l_1st=l
            modules.append(nn.Linear(l, len(self.idx2target)))
            self.model = nn.Sequential(*modules)            
        elif self.task=='classification':
            modules = []
            l_1st=nBits
            for l,do in zip(self.DNNlayers,self.dropout):
                modules.append(nn.Linear(l_1st, l))
                modules.append(self.actF)
                modules.append(nn.LayerNorm(l))
                modules.append(nn.Dropout(float(do)))
                l_1st=l
            modules.append(nn.Linear(l, len(self.idx2target)*2))
            self.model = nn.Sequential(*modules)            

    def forward(self, x):
        tmp = self.model(x)
        if self.task=='regression':
            tmp=tmp
        elif self.task=='classification':
            tmp=tmp.reshape(x.size(0),2,len(self.idx2target))            
        return tmp

    def shared_step(self, batch):
        descriptors, true_val = batch[0], batch[1]
        true_val=torch.nan_to_num(true_val, nan=-1)
        out = self.forward(descriptors)
        indexes_idx=torch.where(true_val!=-1)
        indexes_assays = [self.idx2target[i] for i in indexes_idx[1].cpu().numpy()]
        indexes_comp = [self.idx2compid[i] for i in indexes_idx[0].cpu().numpy()]
        if self.task=='regression':
            out=out[indexes_idx]
            loss = F.mse_loss(out, true_val[indexes_idx])
        elif self.task=='classification':
            #weights= torch.tensor([[0.5,1.0]])
            loss = nn.CrossEntropyLoss(ignore_index=-1)(out, true_val.long())
            out=torch.nn.functional.softmax(out, dim=1, dtype=None)
            out=out[indexes_idx[0],:,indexes_idx[1]]
        true_val=true_val[indexes_idx]
        return {'loss': loss, 'pred': out, 'true': true_val,'assays':indexes_assays,'comp':indexes_comp}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_epoch_end(self, outputs):
        if int(self.current_epoch)%self.savestep==0:
            if self.task=='regression':
                pred = torch.cat([x['pred'] for x in outputs])
            elif self.task=='classification':
                pred = torch.cat([x['pred'] for x in outputs])[:,1]
            true = torch.cat([x['true'] for x in outputs])
            assays = [x['assays'] for x in outputs]
            assays = [i for sublist in assays for i in sublist]
            comp = [x['comp'] for x in outputs]
            comp = [i for sublist in comp for i in sublist]
            self.predT=np.concatenate((self.predT, np.round(pred.cpu().numpy(),3)), 0).squeeze()
            self.trueT=np.concatenate((self.trueT, np.round(true.cpu().numpy(),3)), 0).squeeze()
            self.targID=np.concatenate((self.targID,  np.array(assays)), 0).squeeze()
            self.compID=np.concatenate((self.compID, np.array(comp)), 0).squeeze()
            self.NIter=np.concatenate((self.NIter, np.array([self.current_epoch]*len(true)).astype(int)), 0).squeeze()
            np.savetxt('{}.txt'.format(self.FileName),np.stack((self.predT, self.trueT, self.targID, self.compID, self.NIter), 1), fmt='%s')

    def configure_optimizers(self):
        if self.optF=='RAdam':
            optimizer = optim.RAdam(self.parameters(),lr=self.lr)
        if self.optF=='Adam':
            optimizer = optim.RAdam(self.parameters(),lr=self.lr)
        elif self.optF=='Yogi':
            optimizer = optim.Yogi(self.parameters(),lr=self.lr)
        return optimizer


