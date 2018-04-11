import torch
import torch.nn as nn
from torch.autograd import Variable

class VisualAttentionLoss(nn.Module):
    """
    This loss function can be used for training models with visual attention loss. 
    
    It computes:
         total_loss =  policy_loss + value_loss + classification_loss
         
    where classification loss is CrossEntropy between the last predicted class and the ground truth
          policy loss is REINFORCEMENT loss with -log pi * (reward - value)
          value loss is ACTOR CRITIC loss with MSE between expected reward and actual reward
    """
    
    def __init__(self, gamma): 
        super(VisualAttentionLoss, self).__init__()
        
        # Parameters
        self.gamma = gamma
        
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
        output      : tuple 
                          class_scores        :   torch.FloatTensor
                                                  Predicted classes as tensor with shape [B x C]
                          saved_ln_pis        :   list(T)<torch.FloatTensor>
                                                  Negative saved log propabilities of the taken actions as torch.FloatTensor with shape [B x 1]
                          saved_baselines     :   list(T)<torch.FloatTensor>
                                                  Saved baseline values of the predicted rewards per timestep as torch.FloatTensor with Shape [B x 1]
                          rewards             :   torch.FloatTensor 
                                                  Rewards with shape [B x T]
        labels      : torch.LongTensor
                      Ground truth with shape [B]

        Return
        ------                      
        loss    :   float
        """
        class_scores, saved_baselines, saved_ln_pis, rewards = output
        
        # Loss
        criterion = nn.CrossEntropyLoss()
        classification_loss = criterion(class_scores, labels)
        
        # Compute rewards
        pred = class_scores.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred))
        rewards = self.compute_rewards(rewards, correct)
        
        # Compute policy loss
        _policy_loss, _value_loss = self.reinforce_loss(saved_ln_pis, saved_baselines, rewards)
    
        # Total loss
        total_loss =  _policy_loss + _value_loss + classification_loss
        
        #Update state
        self.__policy_loss = _policy_loss.data
        self.__classification_loss = classification_loss.data
        self.__value_loss = _value_loss.data
            
        return total_loss
       

    def reinforce_loss(self, saved_ln_pis, saved_baselines, rewards):
        """
        Computes the reinforce loss
        
        Arguments
        ---------
        saved_ln_pis        : list(T)<torch.FloatTensor>
                              Negative saved log propabilities of the taken actions as torch.FloatTensor with shape [B x 1]
        saved_baselines     : list(T)<torch.FloatTensor>
                              Saved baseline values of the predicted rewards per timestep as torch.FloatTensor with Shape [B x 1]
        rewards             : torch.FloatTensor 
                              Rewards with shape [B x T]
        
        Return
        ------
        policy_loss     : float
                          REINFORCE loss
        value_loss      : float
                          MSE loss
        """
        if len(saved_ln_pis) == 0:
            return Variable(torch.FloatTensor(1).zero_()), Variable(torch.FloatTensor(1).zero_())
        
        # Batch-size are called episodes in reinforcement learning
        bs = rewards.size(0)
        t = rewards.size(1)
        
        policy_losses = []
        value_losses = []
        # Loop over all timesteps and compute losses
        for ln_pi_t, b_t, r_t in zip(saved_ln_pis, saved_baselines, rewards.t()):
            exp_b_t = b_t.sum(0) / bs
            value_losses.append((exp_b_t - Variable(r_t.unsqueeze(1)))**2)  
            policy_losses.append(-ln_pi_t * (Variable(r_t.unsqueeze(1) - exp_b_t.data)))
                       
        # Sum losses up over timesteps and episodes and divide by episodes
        return torch.cat(policy_losses, dim=1).sum() / (bs * t), torch.cat(value_losses, dim=1).sum() / (bs * t)


    def compute_rewards(self, model_rewards, classification_reward):
        """
        Computes the final rewards by combining collected rewards with classification reward
        
        Arguments
        ---------
        model_rewards           :   list(T)<torch.FloatTensor>
                                    A list containing the rewards for every timestep as torch.FloatTensor with shape [B x 1].
        classification_reward   :   torch.ByteTensor
                                    A byte tensor with shape [B x 1] holding the classification result e {0,1}
        
        Return
        ------
        rewards         : torch.FloatTensor
                          The final decayed rewards as a Tensor with shape [B x T]
        
        """
        if self.gamma > 0.:
            R = classification_reward.float()
            rewards = []
            for r in model_rewards[::-1]:
                R = self.gamma * R
                rewards.insert(0, r + R)       
            rewards = torch.cat(rewards, dim=1) 
        else:
            model_rewards[-1] += classification_reward.float()
            rewards = torch.cat(model_rewards, dim=1) 
        return rewards