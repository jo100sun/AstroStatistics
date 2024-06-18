import module.dataloader
import yaml
import torch as th

config = yaml.safe_load(open("config.yaml"))
dl = module.dataloader.GalaxyDataloader(config)

device = 'cuda'

import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision as tv
import os
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class myNet(th.nn.Module):

    class AttentionLayer(th.nn.Module):
        def __init__(self,embed_dim,num_heads=1,dropout=0.1):
            super().__init__()
            self.mha = th.nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
            self.norm1 = th.nn.LayerNorm(embed_dim)
            self.lin = th.nn.Linear(embed_dim,embed_dim)
            self.norm2 = th.nn.LayerNorm(embed_dim)
            
        def forward(self,x):
            x = x + self.mha(x,x,x)[0]
            x = self.norm1(x)
            x = x + self.lin(x)
            x = self.norm2(th.relu(x))
            return x

    def __init__(self, len_img, nout, patch_len, patch_embed_dim, final_hidden_size, n_heads, dropout, n_blocks):
        super().__init__()
        self.len_img = len_img
        self.patch_len = patch_len
        self.len_patches = len_img//patch_len
        self.patch_embed_dim = patch_embed_dim
        self.lin_projection = th.nn.Linear(3*patch_len*patch_len,patch_embed_dim)
        self.pos_embedding_token = th.nn.Parameter(th.randn(1, self.len_patches*self.len_patches, patch_embed_dim))
        self.blocks  = th.nn.ModuleList(myNet.AttentionLayer(patch_embed_dim,n_heads,dropout) for _ in range(n_blocks))
        self.final1 = th.nn.Sequential(th.nn.Linear(patch_embed_dim,8),th.nn.Dropout(dropout),th.nn.ReLU())
        self.final2 = th.nn.Sequential(th.nn.Linear(8*self.len_patches*self.len_patches, final_hidden_size),
                                         th.nn.GELU(), th.nn.Dropout(dropout), 
                                         th.nn.Linear(final_hidden_size, nout))
        self.softmax = th.nn.Softmax(dim = 1)
        self.dropout = th.nn.Dropout(dropout)

    def forward(self,x):
        x = self.lin_projection(x)
        x = self.dropout(x)
        x = x + self.pos_embedding_token
        for b in self.blocks:
            x = b(x)
        x = self.final1(x)
        x = x.view(-1,8*self.len_patches*self.len_patches)
        x = self.final2(x)
        x = self.softmax(x)
        return x
    
net = myNet(len_img=config['img_size']+2*config['pad'], nout=5, patch_len=config["patch_len"], patch_embed_dim=256, final_hidden_size=512, n_heads=4, dropout=0.3, n_blocks=4).to(device)
if os.path.exists(f'{config["path"]}/model.pth'):
    net.load_state_dict(th.load(f'{config["path"]}/model.pth'))
    print(f'weight loaded from {config["path"]}/model.pth')

nepochs = 30
opt= th.optim.AdamW(params=net.parameters(),lr=0.0003)
lossfn=th.nn.CrossEntropyLoss()
bsz = 256

best = 1e10
for epoch in range(nepochs):
    trainloss,corr,n=0,0,0
    net.train() # This is too simple problem so it seems dropout and layernorm just disturbs training
    for x, y in dl.trainloader(device):
        opt.zero_grad()
        out=net(x)
        #print((y==0).sum(),(y==1).sum(),(y==2).sum(),(y==3).sum(),(y==4).sum())
        loss = lossfn(out,y) # I just simply calculate loss for all sequence including empty space. model should output 11 for empty space.
        loss.backward()
        opt.step()
        trainloss+=loss.item()*len(out)
        corr += (y==out.argmax(dim=1)).sum().item()
        n += len(out)
    print("Train loss : ", trainloss/n, "Accuracy : ",corr/n)
    with th.no_grad():
        net.eval()
        testloss,corr,n=0,0,0
        all_outs, all_labels = th.tensor([]),th.tensor([])
        for x, y in dl.testloader(device):
            out=net(x)
            loss = lossfn(out,y)
            testloss+=loss.item()*len(out)
            corr += (y==out.argmax(dim=1)).sum().item()
            n += len(out)
            all_labels = th.cat([all_labels,y.cpu()],dim=0)
            all_outs = th.cat([all_outs,out.cpu().argmax(dim=1)],dim=0)
        print(" Test loss : ", testloss/n, "Accuracy : ",corr/n) # Accuracy here is average number of correct answer for sequence.
        if testloss/n < best:
            th.save(net.state_dict(),f'{config["path"]}/model.pth')
            print("best model saved")
            best = testloss/n
            cf = confusion_matrix(all_labels,all_outs)
            ConfusionMatrixDisplay(cf).plot().figure_.savefig(f'{config["path"]}/confusion.png')
