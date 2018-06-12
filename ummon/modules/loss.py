#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:21:57 2018

@author: matthias
"""

import torch.nn as nn
import ummon.functionals.visualattention as va

class VisualAttentionLoss(nn.CrossEntropyLoss):
    """
    This loss function can be used for training models with visual attention loss. 
    
    It computes:
         total_loss =  policy_loss + value_loss + classification_loss
         
    where classification loss is CrossEntropy between the last predicted class and the ground truth
          policy loss is REINFORCEMENT loss with -log pi * (reward - value)
          value loss is ACTOR CRITIC loss with MSE between expected reward and actual reward
    """
    
    def __init__(self, model, gamma, size_average = False): 
        super(VisualAttentionLoss, self).__init__()
        
        # Parameters
        self.gamma = gamma
        self.model = model
        self.size_average = size_average
        
        # Save losses for log
        self.__policy_loss = None
        self.__classification_loss = None
        self.__value_loss = None
        
    def __repr__(self):
        return "VisualAttentionLoss():\n\tPolicy: {}, CE: {}, Val: {}".format(
                                self.__policy_loss[0] if self.__policy_loss is not None else "-",
                                self.__classification_loss[0] if self.__classification_loss is not None else "-",
                                self.__value_loss[0] if self.__value_loss is not None else "-")
    
    def forward(self, output, labels):
        output = (output, self.model.policy.saved_baselines, self.model.policy.saved_ln_pis, self.model.policy.rewards)
        classification_loss, _policy_loss, _value_loss = va.visual_attention_loss(output, labels, self.gamma, self.size_average)
        
        # Total loss
        total_loss =  classification_loss + _policy_loss + _value_loss 
        
        #Update state
        self.__policy_loss = _policy_loss.data
        self.__classification_loss = classification_loss.data
        self.__value_loss = _value_loss.data
            
        return total_loss
   
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNSNNLoss(nn.Module):
    """
    This loss function can be used for training deep metric models of size 2. This means 
    that two samples can be computed the same time like common in Siamese Nets.
    
    [1] http://cs231n.stanford.edu/reports/2016/pdfs/258_Report.pdf
    
    """

    def __init__(self, margin=2.0, size_average = False):
        super(ANNSNNLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, preds, labels):
        """
        Computes the contrastive loss
        
        Parameters
        -----------
        * preds (tuple<torch.Tensor>) with shape (B x F, B x F)
        * labels (list) with len B 
        
        """
        
        output1, output2 = preds[0],preds[1]                 # preds: 2 x B x F  => B x F, B x F
        labels_left, labels_right = labels[0], labels[1]     # labels ([B], [B]) => [B], [B]
        label = (labels_left != labels_right).float()        # [B] <float>

        euclidean_distance = F.pairwise_distance(output1, output2)   # B x B
        assert euclidean_distance.size(0) == output1.size(0)

        loss_contrastive = torch.sum((1 - label) * .5 * torch.pow(euclidean_distance, 2) + label * .5 * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if self.size_average == False:
            return loss_contrastive                              # <float>
        else:
            return loss_contrastive / output1.size(0)            # <float>
