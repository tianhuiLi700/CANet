import numpy as np
import torch
from typing import Iterable
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_confusion_matrix(trues, preds):
    labels = [0, 1, 2, 3, 4]
    conf_matrix = confusion_matrix(trues, preds,labels=labels)
    return conf_matrix


def plot_confusion_matrix(conf_matrix):
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1, 2, 3, 4]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig('heatmap_confusion_matrix.jpg')
    plt.show()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum(1).mean()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device,
                    epoch: int,
                    optimimzer):
    model.train()
    loss_all = 0
    iterator = tqdm(data_loader, desc='Training (Epoch %d)'%epoch)

    for idx, (samples, targets) in enumerate(iterator):
        samples = samples.to(device).float()
        pred = model(samples)

        # loss = F.mse_loss(pred, targets['pred'].float().to(device))
        loss = sigmoid_focal_loss(pred[..., None], targets['err'].float().to(device).unsqueeze(-1))
        optimimzer.zero_grad()
        loss.backward()
        optimimzer.step()
        loss_all += loss.detach().cpu().item()
        iterator.set_postfix_str('ave loss={:.6f}'.format(loss_all / (idx+1)))
    return loss_all / (idx+1)



def eval_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device,
                    epoch: int,
                    optimimzer):
    model.eval()
    loss_all = 0
    all_acc = np.asarray([0, 0, 0, 0])
    iterator = tqdm(data_loader, desc='Eval     (Epoch %d)'%epoch)
    test_pred = []
    test_gts = []
    with torch.no_grad():
        for idx, (samples, targets) in enumerate(iterator):
            samples = samples.to(device).float()
            pred = model(samples).sigmoid() > 0.5
            pred = pred.int()
            gt = targets['err'].unsqueeze(-1).to(device)
            pred = pred.detach().cpu().numpy()
            gt = gt[..., 0].detach().cpu().numpy()

            test_pred.extend(pred)
            test_gts.extend(gt)

        sklearn_precision = precision_score(test_gts, test_pred, average='micro')
        sklearn_accuracy = accuracy_score(test_gts, test_pred)
        sklearn_recall = recall_score(test_gts, test_pred, average='micro')
        sklearn_f1 = f1_score(test_gts, test_pred, average='micro')
        report = classification_report(test_gts, test_pred)
        print('\n', classification_report(test_gts, test_pred))
        list_index = [*range(0, len(test_gts) * 4)]
        list_index = np.asarray([i % 4 + 1 for i in list_index])
        gt = np.asarray(test_gts)
        pred = np.asarray(test_pred)
        gt = gt.flatten() * list_index
        pred = pred.flatten() * list_index
        conf_matrix = get_confusion_matrix(gt, pred)
        print(conf_matrix)
        # plot_confusion_matrix(conf_matrix)
        print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy,
                                                                                                  sklearn_precision,
                                                                                                  sklearn_recall,
                                                                                                  sklearn_f1))
    return sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1, report
