from .base import *
import torch
import numpy as np
import torch.functional as F

# Get index of class with max probability
def classify(output, loss_function = None, logger = None):
    """
    Return
    ------
    classes (torch.LongTensor) - Shape [B x 1]
    """
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
    
    if loss_function is not None:
        # Evaluate non-linearity in case of a combined loss-function like CrossEntropy
        if isinstance(loss_function, torch.nn.BCEWithLogitsLoss):
            output = torch.sigmoid(output).data
        
        if isinstance(loss_function, torch.nn.CrossEntropyLoss):
            output = F.softmax(output, dim=1).data

    # Case single output neurons (e.g. one-class-svm sign(output))
    if (output.dim() > 1 and output.size(1) == 1) or output.dim() == 1:
        # makes zeroes negative
        classes = (output - 1e-12).sign().long()  
            
    # One-Hot-Encoding
    if (output.dim() > 1 and output.size(1) > 1):
        classes = output.max(1, keepdim=True)[1]

    return classes


def compute_accuracy(classes, targets):
    if not isinstance(targets, torch.Tensor):
        targets = targets.y

    assert targets.shape[0] == classes.shape[0]
            
    # Classification one-hot coded targets are first converted in class labels
    if targets.dim() > 1 and targets.size(1) > 1:
        targets = targets.max(1, keepdim=True)[1]
    
    if not isinstance(targets, torch.LongTensor):
        targets = targets.long()
    
    # number of correctly classified examples
    correct = classes.eq(targets.view_as(classes))
    sum_correct = correct.sum()
    
    if type(sum_correct) == torch.Tensor:
        sum_correct = sum_correct.item()
    
    # accuracy
    accuracy = sum_correct / len(targets)
    return accuracy

class Accuracy(OnlineMetric):

    def __call__(self, output, labels):
        classes = classify(output.to('cpu'))
        acc = compute_accuracy(classes, labels.to('cpu'))
        return acc * 100

    def __repr__(self):
        return "acc"
