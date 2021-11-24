#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
import copy
#matplotlib.use('Agg')

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.NLLLoss()
        self.ldr = self.train_val_test(dataset, list(idxs))          

    def train_val_test(self, dataset, idxs):
        # split train, and test
        data = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        #val = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=True)
        # test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        return data

    def update_weights(self, net):
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9)
        #optimizer = optim.Adam([var1, var2], lr = 0.0001)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)  # Adam 优化器
        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()          
                    images, labels = autograd.Variable(images),\
                                    autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu == -1:
                    loss = loss.cpu()
                # if self.args.verbose and batch_idx % 10 == 0:
                #    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #               100. * batch_idx / len(self.ldr_train), loss.data.item()))
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            acc, _, = self.test(net)
            # print('\nLabels:', labels.data, y_pred)
            # if (iter+1)%10 == 0: 
            #    print("local epoch:", iter)
            #    print("acc: {}".format(acc))
            epoch_acc.append(acc)
        avg_loss = sum(epoch_loss)/len(epoch_loss)
        avg_acc = sum(epoch_acc)/len(epoch_acc)
        w = net.state_dict()         
        return w, avg_loss ,avg_acc
          

    def test(self, net):
        loss = 0
        log_probs = []
        labels = []
        for batch_idx, (images, labels) in enumerate(self.ldr):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
        if self.args.gpu != -1:
            loss = loss.cpu()
            log_probs = log_probs.cpu()
            labels = labels.cpu()
        y_pred = np.argmax(log_probs.data, axis=1)
        acc = metrics.accuracy_score(y_true=labels.data, y_pred=y_pred)
        loss = loss.data.item()         
        return acc, loss
