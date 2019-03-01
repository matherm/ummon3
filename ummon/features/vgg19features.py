#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3') 
sys.path.insert(0,'../ummon3')     
#############################################################################################

import torch
import numpy as np
import torch.nn as nn
from ummon.modules.vgg19 import VGG19
from ummon.modules.gram import GramMatrix
from torchvision.models import vgg19
from pathlib import Path
import os

__all__ = ['VGG19Features']

class VGG19Features():
    """
    Extracts features from VGG19
    
    Usage
    ======
            transform = VGG(features="pool4")
            transform(tensor)
            
            OR
            
            vgg = VGG19Features(features="pool4")
            my_transforms = transforms.Compose([transforms.ToTensor(), vgg])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)
    
    """
    def __init__(self, features = "pool4", gram = False, triangular=False, cuda = False, pretrained = True, gram_diagonal = False, gram_diagonal_squared = False, selected_feat = None):
        """
        Parameters
        ----------
            *features (list or str):  Features: ['pool4', 'pool1', 'pool2', 'relu1_1', 'pool3', 'relu1_2']
            *gram (bool) : compute gram
            *gram_diagonal (bool) : compute only gram diagonal i.e. the scalar product of a feature map
            *gram_diagonal_squared (bool) : compute only the squared gram diagonal i.e. the l4 norm of a feature map
            *triangular (bool) : triangular matrix 
            *cuda (bool) : compute with cuda
            *pretrained (bool) : use pretrained vgg19 net
            *selected_feat (list of int):  contains feature correlations(entries of the Gram-Matrix) selected_feat: ['23', '25', '44']
            v23,v23 || v23,v25 || v23,v44 || v25,v25 || v25,v44 || v27,v44 || v44,v44
        """
        self.features = features
        self.gram = gram
        self.cuda = cuda
        self.triangular = triangular
        self.pretrained = pretrained
        self.gram_diagonal = gram_diagonal
        self.gram_diagonal_squared = gram_diagonal_squared
        self.selected_feat = selected_feat

        # Original PyTorch VGG19
        self.model = vgg19(pretrained=self.pretrained).features

        # Custom VGG19 with better Layer naming convention
        self.vgg19 = VGG19()

        if pretrained:
            # Copy weights
            self.copy_weights_(self.model, self.vgg19)
        else:
            self.vgg19.apply(self.random_weight_init)

        # Cuda
        self.vgg19 =  self.vgg19.cuda() if self.cuda and torch.cuda.is_available() else self.vgg19
        
        #Eval mode
        self.vgg19.eval()
        
        if type(self.features) is not list:
            self.features  = [self.features]
            

    def random_weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data,0)

            
    def copy_weights_(self, model_source, model_dest):
        for i, k in enumerate(model_dest._modules):
            if isinstance(model_dest._modules[k], nn.Conv2d):
                model_dest._modules[k].weight = model_source[i].weight
                model_dest._modules[k].bias = model_source[i].bias
        

    def __call__(self, x):
        if x.dim != 3 and x.size(0) != 3:
            raise ValueError("Shape of input should be (3, x, x)")
        x = x.unsqueeze(0)
  
        if self.cuda and torch.cuda.is_available():
            x = x.cuda()
            
        result = []
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            if name in self.features:
                y = x
                if self.gram:
                    y = GramMatrix()(y)
                    if self.triangular:
                        y = y[:, np.triu_indices(x.shape[1])[0], np.triu_indices(x.shape[1])[1]]
                if (self.gram_diagonal or self.gram_diagonal_squared) and (self.selected_feat is None):
                    y = y.view(y.size(0), y.size(1), 1, y.size(2) * y.size(3))
                    gram_diag = None
                    for b in range(y.size(0)):
                        if self.gram_diagonal_squared:
                            z = torch.bmm(y[b]*y[b], (y[b]*y[b]).transpose(2, 1))
                        else:
                            z = torch.bmm(y[b], y[b].transpose(2, 1))
                        if isinstance(gram_diag, torch.Tensor):
                            gram_diag = torch.cat(gram_diag, z)
                        else:
                            gram_diag = z
                    y = torch.squeeze(gram_diag)
                    y = torch.unsqueeze(y, 0)
                if not(self.gram_diagonal and self.gram_diagonal_squared and self.gram) and (self.selected_feat is not None):
                    #Normalize with max
                    y = y/y.max()
                    b, c, h, w = y.size()
                    res=[]
                    for feat in self.selected_feat:
                        for feat2 in self.selected_feat:
                            if feat is not feat2 and feat2 > feat:
                                k = y[:, feat, : , :]
                                F = k.view(b, 1, h * w)
                                k2 = y[:, feat2, :, :]
                                F2 = k2.view(b, 1, h * w)
                                G = torch.bmm(F, F2.transpose(1, 2))
                                G.div_(h * w)
                                res.append(G)

                    res = torch.cat(res, dim=1)
                    y = res


                if len(self.features) == 1:
                    # get first item, preserve shape
                    result = y.cpu()[0].detach()
                    break;
                else:
                    # get first item, not perserving shape
                    result.append(y.cpu().view(-1).detach())
                    if len(result) == len(self.features):
                        break;
        
        if len(self.features) == 1:
            return result
        else:    
            return torch.cat(result).cpu()
