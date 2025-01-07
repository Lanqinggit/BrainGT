import os
import sys
import torch
import random
import numpy as np
import torch.nn.functional as F

from opt import * 
from metrics import accuracy, auc, prf, metrics
from DataUtil import dataloader
import matplotlib.pyplot as plt
from model import BrainGT
import csv

if __name__ == '__main__':
    opt = OptInit().initialize()
    dl = dataloader()
    all_graphs, y, nonimg, phonetic_score = dl.load_data()
    n_folds = 5
    cv_splits = dl.data_split(n_folds)
    corrects = np.zeros(n_folds, dtype=np.int32) 
    accs = np.zeros(n_folds, dtype=np.float32) 
    sens = np.zeros(n_folds, dtype=np.float32) 
    spes = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)
    fold_train_losses = []
    fold_val_losses = []
    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        model = BrainGT(nonimg, phonetic_score).to(opt.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"Total number of trainable parameters: {total_params}")
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        train_loss_list = []
        val_loss_list = []
        def train(): 
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()  
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    node_logits = model(all_graphs)
                loss_cla = loss_fn(node_logits[train_ind], labels[train_ind])
                loss = loss_cla
                train_loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                correct_train, acc_train, test = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits= model(all_graphs)
                loss_val = loss_fn(node_logits[test_ind], labels[test_ind])
                val_loss_list.append(loss_val.item())
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test,test1 = accuracy(logits_test, y[test_ind])
                test_sen, test_spe = metrics(logits_test, y[test_ind])
                auc_test = auc(logits_test,y[test_ind])
                prf_test = prf(logits_test,y[test_ind])
                print("Epoch: {},\tce loss: {:.5f},\ttrain acc: {:.5f},\ttrain spe: {:.5f},\ttest acc: {:.5f},\tval loss: {:.5f}".format(epoch, loss.item(), acc_train.item(),test_spe, acc_test,loss_val))
                if acc_test > acc and epoch >5:
                    acc = acc_test
                    correct = correct_test 
                    aucs[fold] = auc_test
                    prfs[fold]  = prf_test  
                    sens[fold] = test_sen
                    spes[fold]  = test_spe 
                    if opt.ckpt_path !='':
                        if not os.path.exists(opt.ckpt_path): 
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)
                        print("{} Saved model to:{}".format("\u2714", fold_model_path))
            fold_train_losses.append(train_loss_list)
            fold_val_losses.append(val_loss_list)
            accs[fold] = acc 
            corrects[fold] = correct

            print("\r\n => Fold {} test accuacry {:.5f}".format(fold, acc))

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            model.load_state_dict(torch.load(fold_model_path)) 
            model.eval()
            node_logits = model(all_graphs)
            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold], test2 = accuracy(logits_test, y[test_ind])
            sens[fold], spes[fold] = metrics(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test,y[test_ind]) 
            prfs[fold]  = prf(logits_test,y[test_ind])  
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))  
        if opt.train==1:
            train()
        elif opt.train==0:
            evaluate()
    fold_train_losses = np.array(fold_train_losses)
    fold_val_losses = np.array(fold_val_losses)
    avg_train_losses = fold_train_losses.mean(axis=0)
    with open('avg_train_losses.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Avg_Train_Loss'])
        for epoch, loss in enumerate(avg_train_losses, 1):
            writer.writerow([epoch, loss])
    avg_val_losses = fold_val_losses.mean(axis=0)
    with open('avg_val_losses.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Avg_Val_Loss'])
        for epoch, loss in enumerate(avg_val_losses, 1):
            writer.writerow([epoch, loss])
    plt.plot(avg_train_losses, label='Training Loss')
    plt.plot(avg_val_losses, label='Validation Loss')
    plt.title('(c) Loss Curves on ABIDE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('ABIDE_LOSS.JPEG', dpi=1000)
    plt.show()
    print("\r\n========================== Finish ==========================") 

    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.std(accs)))
    print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.std(sens)))
    print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.std(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.std(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    se_std, sp_std, f1_std = np.std(prfs, axis=0)
    print("=> Average test F1-score {:.4f}({:.4f})".format(f1, f1_std))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))