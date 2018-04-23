#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:21:57 2018

@author: matthias
"""

import torch.nn as nn
import ummon.functionals.visualattention as va

class VisualAttentionLoss(nn.Module):
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
   
