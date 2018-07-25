import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Implementation of REINFORCE by Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
"""

def reinforce_loss(saved_ln_pis, saved_baselines, rewards, size_average = False):
    """
    Computes the reinforce loss.
    
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
    assert len(saved_ln_pis) == len(saved_baselines) == len(rewards[0])

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
    if size_average:
        return torch.cat(policy_losses, dim=1).sum() / (bs * t), torch.cat(value_losses, dim=1).sum() / (bs * t)
    else:
        return torch.cat(policy_losses, dim=1).sum() / (t), torch.cat(value_losses, dim=1).sum() / (t)


def decay_rewards(model_rewards, rewards, gamma):
    """
    Computes the final rewards by decaying a final reward with decay rate gamma.
    
    Example
    -------
    model_rewards = [0, 1, 2, 3, 4]
    rewards = 10
    gamma = 0.9
    result = [0.9(0.9(0.9(0.9*10))), 1 + 0.9(0.9(0.9*10)), 2 + 0.9(0.9*10), 3 + 0.9*10, 4 + 10]
        
    Arguments
    ---------
    model_rewards           :   list(T)<torch.FloatTensor>
                                A list containing the rewards for every timestep as torch.FloatTensor with shape [B x 1].
    reward   :                  torch.Tensor
                                A byte tensor with shape [B x 1]
    
    Return
    ------
    rewards         : torch.FloatTensor
                      The final decayed rewards as a Tensor with shape [B x T]
    
    """
    if gamma == 0:
        model_rewards[-1] += rewards.float()
        rewards = torch.cat(model_rewards, dim=1) 
    else:
        R = rewards.float()
        rewards = [model_rewards[-1] + R]
        for r in model_rewards[::-1][1:]:
            R = gamma * R
            rewards.insert(0, r + R)       
        rewards = torch.cat(rewards, dim=1) 
    return rewards
