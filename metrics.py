import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy.special import softmax


def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    test = np.argmax(preds, 1)
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction), test


def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out


def prf(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    pred_lab= np.argmax(preds, 1)
    p,r,f,s  = precision_recall_fscore_support(labels, pred_lab, average='binary')
    return [p,r,f]


def numeric_score(pred, gt):
    FP = np.float64(np.sum((pred == 1) & (gt == 0)))
    FN = np.float64(np.sum((pred == 0) & (gt == 1)))
    TP = np.float64(np.sum((pred == 1) & (gt == 1)))
    TN = np.float64(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN


def metrics(preds, labels):
    preds = np.argmax(preds, 1) 
    FP, FN, TP, TN = numeric_score(preds, labels)
    sen = TP / (TP + FN + 1e-10)  # recall sensitivity
    spe = TN / (TN + FP + 1e-10)
    return sen, spe

