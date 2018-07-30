#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:21:57 2018

@author: matthias
"""

import torch.nn as nn
import ummon.functionals.reinforcement as reinforce


__all__ = [ 'VisualAttentionLoss', 'Contrastive', 'TriContrastive']

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
        
        assert hasattr(model, "policy") 
        assert hasattr(model.policy, "saved_ln_pis")
        assert hasattr(model.policy, "rewards")
        
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
        output (torch.FloatTensor) : class_scores [B]
        labels      : torch.LongTensor
                      Ground truth with shape [B]
    
        Return
        ------                      
        loss    :   float
        """
        saved_baselines, saved_ln_pis, rewards = self.model.policy.saved_baselines, self.model.policy.saved_ln_pis, self.model.policy.rewards
        
        # ASSERTIONS
        assert labels.dtype == torch.long
        assert output.dtype == torch.float32
        assert saved_baselines[0].dtype == torch.float32
        assert saved_ln_pis[0].dtype == torch.float32
        assert rewards[0].dtype == torch.float32
        
        # GOTO CPU
        output = output.cpu()
        labels = labels.cpu()
        if saved_baselines[0].is_cuda:
            saved_baselines = [sb.cpu() for sb in saved_baselines]
        if saved_ln_pis[0].is_cuda:
            saved_ln_pis = [ln_pi.cpu() for ln_pi in saved_ln_pis]
        if rewards[0].is_cuda:
            rewards = [r.cpu() for r in rewards]
                
        # CLASSIFIER
        if labels.dim() > 1:
            labels = labels.view(labels.size(0))
        criterion = nn.CrossEntropyLoss(size_average = self.size_average)
        classification_loss = criterion(output, labels).cpu()
        
        # REINFORCE LOSS
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred))
        rewards = reinforce.decay_rewards(rewards, correct, self.gamma)
        _policy_loss , _value_loss  = reinforce.reinforce_loss(saved_ln_pis, saved_baselines, rewards, self.size_average)
        
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
        * pdist (<torch.Tensor>) with shape B
        * labels (2-tuple) with len 
        
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
        
        
class TriContrastive(nn.Module):
    """
    This loss function can be used for training deep metric models of size 3. 
    This means that three samples can be computed the same time like common in Triplet Nets.
    
    [1] https://omoindrot.github.io/triplet-loss
    
    """

    def __init__(self, margin=2.0, size_average = False):
        super(TriContrastive, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, pdist, labels):
        """
        Computes the contrastive loss
        
        Parameters
        -----------
        * pdist (3-tuple) with len B à (o1, o2, o3)
        * labels (3-tuple) with len B à (y1, y2, y3)
        
        """
        label_anchor, label_pos, label_neg = labels[0], labels[1], labels[2] # labels ([B], [B]) => [B], [B]
        out_anchor, out_pos, out_neg = pdist[0], pdist[1], pdist[2]
        B = labels[0].size(0)
        assert pdist[0].size(0) == labels[0].size(0)
        assert label_anchor[0] == label_pos[0] and label_anchor[0] != label_neg[0]
        
        dist_pos = F.pairwise_distance(out_anchor, out_pos)        # B
        dist_neg = F.pairwise_distance(out_anchor, out_neg)        # B
        
        loss_contrastive = torch.sum(F.relu(self.margin + dist_pos - dist_neg))    # B
        
        if self.size_average == False:
            return loss_contrastive                              # <float>
        else:
            return loss_contrastive / B                          # <float>