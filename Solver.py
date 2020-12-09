# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import time
import os
import gc
import sys
from torch.optim import lr_scheduler
import h5py
import utils
import torch.nn.functional as F
from sklearn import datasets, svm, metrics
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default=None, help='Please set the root folder of datasets')
opt, _ = parser.parse_known_args()


class Solver:
    # Setting basic parameters during Class init...

    def __init__(self, net, model_confs, solver_confs):
        self.mode = solver_confs.mode # 'trainval_action' or 'extract_action_feas' or 'trainval_activity' or 'end_to_end'
        # data args
        self.data_loaders = solver_confs.data_loaders
        self.data_sizes = solver_confs.data_sizes
        self.num_frames = model_confs.num_frames
        self.K = model_confs.num_players
        self.T = None

        # net args
        self.net = net

        # model training args
        self.gpu = solver_confs.gpu
        self.num_epochs = solver_confs.num_epochs
        self.optimizer = solver_confs.optimizer
        self.criterion = solver_confs.criterion
        self.scheduler = solver_confs.exp_lr_scheduler
        self.label_type = solver_confs.label_type
        if opt.save_path:
            self.save_path = opt.save_path
        else:
            self.save_path = os.path.join('./ckpt', solver_confs.dataset_name, solver_confs.stage)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)



    def training(self, inputs, labels, phase):
        # Get the inputs, and wrap them in Variable
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        
        # Forward
        _, outputs = self.net(inputs)
        
        result = outputs.data
        if phase == 'test':
            _, preds = torch.max(torch.mean(outputs.data, 0).view(1, -1), 1)
            if self.mode == 'end_to_end':
                result = result.view(result.size(0)//self.T,self.T,-1) #result: (batch, T, num_classes)
                #print 'result size:', result.size()
                _, preds = torch.max(torch.mean(result, 1).squeeze(), -1) # preds: (batch,1)
                #print 'preds size:', preds.size()
                preds = torch.t(preds.repeat(10,1)).contiguous().view(-1) # preds: (batch, T)
                #print 'preds size:', preds.size()
                
                #outputs.data = outputs.data.view(outputs.data.size(0)/10,10,-1) #result: (2, 10, num_classes)
                #print 'outputs.data size:', outputs.data.size()
                #outputs.data = torch.mean(outputs.data, 0).squeeze() # preds: (10, num_classes)
                #print 'outputs.data size:', outputs.data.size()
        else:
            _, preds = torch.max(result, 1)

        # print outputs, labels
        if self.mode == 'end_to_end':
            labels = labels[::self.K] #### please note that the step is self.num_players, not 10.
        else:
            labels = labels

        loss = self.criterion(outputs, labels)
#         print(loss)
        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            loss.backward()
            self.optimizer.step()    
        
        # statistics
        self.running_loss += loss.item()
        self.running_corrects += torch.sum(preds == labels)
        
        
    def end2end_training(self, inputs, labels, phase):
#         inputs = inputs.view(-1, *inputs.size()[2:])
        labels = labels.view(-1, *labels.size()[2:])
        labels = torch.split(labels, 1, dim=-1)
        action_labels, activity_labels = labels[0].squeeze(), labels[1].squeeze()
#         print(action_labels.size(), activity_labels.size())
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Forward
        _, action_scores, activity_scores, _, _ = self.net(inputs)
        if self.T!=1 and phase=='test':
            action_scores = action_scores.view(-1, self.T, action_scores.size(-1)).mean(dim=1)
            activity_scores = activity_scores.view(-1, self.T, activity_scores.size(-1)).mean(dim=1)
            action_labels = action_labels[::self.T]
            activity_labels = activity_labels[::self.T*self.K] #### please note that the step is self.num_players, not 10.
        else:
            activity_labels = activity_labels[::self.K]
            
        action_preds = torch.argmax(action_scores, 1)
        activity_preds = torch.argmax(activity_scores, 1)

#         print(action_scores.size(), action_labels.size())
        #action_loss = F.cross_entropy(action_scores, action_labels, weight=torch.tensor([1,1,1,1,1,1,1,1,1,0]).float().cuda()) # zero for none.jpg (labeled by '9')

        activity_loss = self.criterion(activity_scores, activity_labels)
        #total_loss = 0*action_loss + activity_loss
        total_loss = activity_loss

        # Backward + optimize(update parameters) only if in training phase
        if phase == 'trainval':
            total_loss.backward()
            self.optimizer.step()    
        
        # statistics
        self.running_loss += total_loss.item()
        self.running_corrects += torch.sum(activity_preds == activity_labels)
        return activity_preds, activity_labels
        
    def train_model(self, phases=['trainval', 'test']):
        best_acc = 0.0
        best_avg_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and evaluate phase
            for phase in phases:
                since = time.time()
                preds = []
                targets = []
                self.T = self.num_frames[phase]
                if phase == 'trainval':
#                     self.scheduler.step()
                    self.net.train()  # Set model to training mode
                else:
                    self.net.eval()  # Set model to evaluate mode

                self.running_loss = 0.0
                self.running_corrects = 0.0

                # Iterate over data.
                for data in self.data_loaders[phase]:
                    # get the inputs
                    inputs = data[0].float()
                    labels = data[1]
                    
                    if self.gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    
                    pred, target = self.end2end_training(inputs, labels, phase)
                    pred, target = pred.view(-1), target.view(-1)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    
                if phase == 'trainval':
                    self.scheduler.step()
                
                if phase=='test':
                    epoch_loss = float(self.running_loss) / ((self.data_sizes[phase]))
                    epoch_acc = float(self.running_corrects) / (self.data_sizes[phase])
                else:
                    epoch_loss = float(self.running_loss) / ((self.data_sizes[phase]) * self.T)
                    epoch_acc = float(self.running_corrects) / (self.data_sizes[phase] * self.T)

                
                ##################
                # added by Mr. Yan
                preds_array = np.asarray(preds, dtype=int).reshape(-1)
                targets_array = np.asarray(targets, dtype=int).reshape(-1)
                epoch_each_acc, epoch_avg_acc = utils.get_avg_acc(preds_array, targets_array)
                
                # display related Info(Loss, Acc, Time, etc.)
                print('Epoch: {} phase: {} Loss: {} Acc: {} Avg_acc: {}'.format(
                    epoch, phase, epoch_loss, epoch_acc, epoch_avg_acc))

                time_elapsed = time.time() - since
                print('Running this epoch in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, 'best_acc_wts.pth'))
                if phase == 'test' and epoch_avg_acc > best_avg_acc:
                    best_avg_acc = epoch_avg_acc
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, 'best_avg_acc_wts.pth'))
        print('Best test Acc: {:4f}'.format(best_acc))
        print('Best test Avg_acc: {:4f}'.format(best_avg_acc))
    
    def evaluate(self):
        self.preds = []
        self.targets = []
        self.T = self.num_frames['test']
        HPIM_f_dict, HAIM_f_dict = {}, {}
        with torch.no_grad():
            for i, data in enumerate(self.data_loaders['test']):
                inputs = data[0].cuda()
                labels = data[-2]
                labels = labels.view(-1, *labels.size()[2:])
                labels = torch.split(labels, 1, dim=-1)
                target = labels[1].squeeze()[0]
                img_paths = data[-1]
                # compute output
                _, _, activity_scores, BIM_f, PIM_f = self.net(inputs, 'test')
                activity_scores = activity_scores.view(self.T, -1).mean(dim=0)
                pred = torch.argmax(activity_scores, -1)
                self.preds.append(pred.cpu())
                self.targets.append(target)
                clip_id = '/'.join(img_paths[0][0].split('/')[4:6])
#                 print(HPIM_f.size())
                HPIM_f_dict[clip_id] = HPIM_f.cpu()
#                 HAIM_f_dict[clip_id] = HAIM_f
                if i==100:
                    utils.save_pkl(HPIM_f_dict, 'HPIM_f_dict')
                    HPIM_f_dict = {}
                    break
#             utils.save_pkl(HPIM_f_dict, 'HPIM_f_dict')
#           utils.save_pkl(HAIM_f_dict, 'HAIM_f_dict')
            self.show()

    def show(self):
        ### show result
        preds = np.asarray(self.preds, dtype=int)
        #preds = label_map(preds)
#         print(self.targets)
        targets = np.asarray(self.targets, dtype=int)
        #labels = label_map(labels)
        preds, targets = preds.reshape(-1,1), targets.reshape(-1,1)
        print("Classification report for classifier \n %s" % (metrics.classification_report(targets, preds)))
        print("Confusion matrix:\n%s" % utils.normlize(metrics.confusion_matrix(targets, preds)))
        print (np.sum(preds == targets), preds.shape, targets.shape)
        print (np.sum(preds == targets) / float(targets.shape[0]))


        # Compute confusion matrix
        cnf_matrix = metrics.confusion_matrix(targets, preds)
        print (cnf_matrix)
        np.set_printoptions(precision=2)
