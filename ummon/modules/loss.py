#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:21:57 2018

@author: matthias
"""

import torch.nn as nn
import ummon.functionals.visualattention as va


__all__ = [ 'VisualAttentionLoss', 'Contrastive' ]

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
        """
        Computes the combined loss of REINFORCE, Actor-Critic and CrossEntropy 
        
        Arguments
        ---------
        output (torch.FloatTensor) : class_scores 
        labels      : torch.LongTensor
                      Ground truth with shape [B]
    
        Return
        ------                      
        loss    :   float
        """
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

class Contrastive(nn.Module):
    """
    This loss function can be used for training deep metric models of size 2. This means 
    that two samples can be computed the same time like common in Siamese Nets.
    
    [1] http://cs231n.stanford.edu/reports/2016/pdfs/258_Report.pdf
    
    """

    def __init__(self, margin=2.0, size_average = False):
        super(Contrastive, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, pdist, labels):
        """
        Computes the contrastive loss
        
        Parameters
        -----------
        * pdist (<torch.Tensor>) with shape (B x B)
        * labels (list) with len B 
        
        """
        assert pdist.size(0) == labels[0].size(0)
        B = labels[0].size(0)
        labels_left, labels_right = labels[0], labels[1]     # labels ([B], [B]) => [B], [B]
        label = (labels_left != labels_right).float()        # [B] <float>

        loss_contrastive = torch.sum((1 - label) * .5 * torch.pow(pdist, 2) + label * .5 * torch.pow(
                torch.clamp(self.margin - pdist, min=0.0), 2))
        if self.size_average == False:
            return loss_contrastive                              # <float>
        else:
            return loss_contrastive / B            # <float>