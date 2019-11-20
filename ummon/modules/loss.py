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
        output = output.to('cpu')
        labels = labels.to('cpu')
        if saved_baselines[0].is_cuda:
            saved_baselines = [sb.to('cpu') for sb in saved_baselines]
        if saved_ln_pis[0].is_cuda:
            saved_ln_pis = [ln_pi.to('cpu') for ln_pi in saved_ln_pis]
        if rewards[0].is_cuda:
            rewards = [r.to('cpu') for r in rewards]
                
        # CLASSIFIER
        if labels.dim() > 1:
            labels = labels.view(labels.size(0))
        criterion = nn.CrossEntropyLoss(size_average = self.size_average)
        classification_loss = criterion(output, labels).to('cpu')
        
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
    [2] https://arxiv.org/pdf/1412.6622.pdf <DEEP METRIC LEARNING USING TRIPLET NETWORK>
    [3] https://arxiv.org/pdf/1503.03832.pdf <FaceNet: A Unified Embedding for Face Recognition and Clustering>
    [4] https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    
    L_euclidean = || dist_a - dist_p || ^ 2 - || dist_a - dist_n || ^ 2 + alpha , referes to [3]
    L_exp = alpha * (exp_dist_p/(exp_dist_p+exp_dist_n))^2, refers to [2]
    L_log = -log(-(dist_a-dist_p)^2/beta + 1 + eps) - log(-N-(dist_a-dist_p)^2/beta + 1 + eps), refers to [4]
    
    Parameters
    -----------
    *alpha (float) specifiying the margin/alpha to use
    *size_average (bool): specifies wheter mini batches should be averaged
    *mode (string) : euclidean | exp | log
    
    """

    def __init__(self, alpha=1.0, size_average = False, mode="euclidean"):
        super(TriContrastive, self).__init__()
        assert mode in ["euclidean", "exp", "log", "expratio"]

        self.alpha = alpha
        self.size_average = size_average
        self.mode = mode

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
        N = out_anchor.size(1)
        assert pdist[0].size(0) == labels[0].size(0)
        assert label_anchor[0] == label_pos[0] and label_anchor[0] != label_neg[0]
        
        if self.mode == "euclidean":
            dist_pos = F.pairwise_distance(out_anchor, out_pos)           # B
            dist_neg = F.pairwise_distance(out_anchor, out_neg)           # B
            loss =  self._euclidean(dist_pos, dist_neg, alpha=self.alpha) # B
            
        if self.mode == "exp":
            dist_pos = F.pairwise_distance(out_anchor, out_pos)             # B
            dist_neg = F.pairwise_distance(out_anchor, out_neg)             # B
            loss =  self._exponential(dist_pos, dist_neg)                   # B
            
        if self.mode == "expratio":
            dist_pos = F.pairwise_distance(out_anchor, out_pos)             # B
            dist_neg = F.pairwise_distance(out_anchor, out_neg)             # B
            loss =  self._expratio(dist_pos, dist_neg)                      # B
          
        if self.mode == "log":
            dist_pos = F.pairwise_distance(F.sigmoid(out_anchor), F.sigmoid(out_pos)) # B
            dist_neg = F.pairwise_distance(F.sigmoid(out_anchor), F.sigmoid(out_neg)) # B
            loss = self._log(dist_pos, dist_neg, N=N, beta=N)               # B
        
        if self.size_average == False:
            return loss                              # <float>
        else:
            return loss / B                          # <float>
        
        
    def _log(self, dist_pos, dist_neg, N, beta, eps=1e-9):
        dist_pos = torch.pow(dist_pos,2) / beta
        dist_neg = (N-torch.pow(dist_neg, 2)) / beta
        dist_pos = -torch.log(-dist_pos + 1 + eps)
        dist_neg = -torch.log(-dist_neg + 1 + eps)
        return torch.sum(dist_pos + dist_pos) 
        
    def _exponential(self, dist_pos, dist_neg):
        # Prepare squared exponential losses
        dist_pos, dist_neg = torch.exp(dist_pos), torch.exp(dist_neg)
        total_dist = dist_pos + dist_neg
        # Normalize losses
        dist_pos = dist_pos / total_dist        
        dist_neg = dist_neg / total_dist        
        # Optimize ratio as L(d_+, d_-) -> 0 iff dist_pis/dist_neg ->
        # || d_+, d_- -1 || ^ 2 = const * d_+^2
        return torch.sum(torch.pow(dist_pos - dist_neg + 1, 2))
    
    def _expratio(self, dist_pos, dist_neg):
        # Prepare squared exponential losses
        dist_pos, dist_neg = torch.exp(-dist_pos), torch.exp(-dist_neg)
        total_dist = dist_pos + dist_neg
        # Normalize losses
        ratio = -torch.log(dist_pos / total_dist)        
        return torch.sum(ratio)
    
    def _euclidean(self, dist_pos, dist_neg, alpha):
        return torch.sum(F.relu(alpha + dist_pos - dist_neg))
        
        
        
        
        