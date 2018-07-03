#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import math
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.modules.activation
from torch.nn import ReLU
from .logger import Logger
import ummon.utils as uu
from .modules.container import Sequential

__all__ = ["Visualizer"]

class Visualizer:
    '''
    This object is used for visualizing what a neural network has learned. Currently,
    only one-branch networks consisting of ummon layers and pytorch activation functions
    are supported. Created by::
    
        vis = Visualizer(model)
    
    where model is a pyTorch network where the properties of single neurons are to be
    visualized. The network with only one branch, either of type ummon.Sequential or
    nn.Sequential.
    
    Visualization methods:
    
    * Finding the n input regions that activate a feature map 'fm' most: get_max_inputs()
    * Saliency map: gradient of the network w.r.t. the input
    * Saliency map: gradient computed by guided back propagation
    
    '''
    def __init__(self, model):
        
        # init internal variables
        self.name = "ummon.Visualizer"
        self._gradient = None
        
        # get names of available activation functions (needed by _find_input_block())
        act_funcs = dir(torch.nn.modules.activation)
        self._act_funcs = act_funcs.copy()
        for i, af in enumerate(act_funcs):
            if af[:2] == '__': # remove internal names
                self._act_funcs.remove(af)
            if af == 'warnings' or af == 'torch':
                self._act_funcs.remove(af)
        
        # check network structure
        if type(model) != Sequential and not isinstance(model, nn.Sequential):
            raise TypeError('Network must consist of only one branch for this method to work.')
        self.model = model
    
    
    # internal function: prepare dataset and check it
    def _prepare_dataset(self, dataset, batch_size):
        if isinstance(dataset, np.ndarray) or uu.istensor(dataset):
            torch_dataset = uu.construct_dataset_from_tuple(Logger(), dataset, train=False)
        if isinstance(dataset, torch.utils.data.Dataset):
            torch_dataset = dataset
        if isinstance(dataset, torch.utils.data.DataLoader):
            dataloader = dataset
            torch_dataset = dataloader.dataset
        else:
            bs = len(torch_dataset) if batch_size == -1 else batch_size
            dataloader = DataLoader(torch_dataset, batch_size=bs, shuffle=False, 
                sampler=None, batch_sampler=None)  
        assert uu.check_precision(torch_dataset, self.model)
        return dataloader
    
    
    # internal function: build subnet to desired layer and get input and output size
    def _build_subnet(self, layer):
        layers = []
        first = True
        found = False
        for key, module in self.model._modules.items():
            if first:
                if module.__class__.__name__ != 'Unflatten': 
                    raise ValueError('Method works only when first layer is Unflatten.')
                self._insize = module.outsize
                first = False
            layers.append((key, module))
            if module.__class__.__name__ not in self._act_funcs:
                self._outsize = module.outsize
            if key == layer:
                found = True
                break
        if not found:
            raise ValueError('Layer {} does not exist in network.'.format(layer))
        self._testnet = Sequential(*layers)
    
    
    # Find the n inputs that maximally activate the units in a feature map
    def _get_max_units(self, fm, n, dataloader):
        '''
        Internal function: Finds the 'n' inputs that maximally activate the units in a 
        feature map.
        
        Arguments:
        
        * fm: feature map index of interest in the current output layer.
        * n: number of inputs desired.
        * dataloader for test data (comes from get_max_inputs())
        
        Output: a list of lists, each entry is a list of the format [ y-position of unit,
        x-position, index of maximally activating input image, activation ]. The list is
        sorted in descending order, starting from the maximum.
        '''
        # init unit activity list
        ua = [[-1, -1, -1, -math.inf] for i in range(n)]
        
        self._testnet.eval() # switch to evaluation mode
        use_cuda = next(self._testnet.parameters()).is_cuda
        for i, data in enumerate(dataloader, 0):
            
            # Get input mini batch
            mini_batch = data
            
            # Handle cuda
            if use_cuda:
                if type(mini_batch) == tuple or type(mini_batch) == list:
                    mini_batch = uu.tensor_tuple_to_cuda(mini_batch)
                else:
                    mini_batch = mini_batch.cuda()
            
            # forward pass through the network
            if type(mini_batch) == tuple or type(mini_batch) == list:
                mini_batch = uu.tensor_tuple_to_variables(mini_batch)
            else:
                mini_batch = Variable(mini_batch)
            outp = (self._testnet(mini_batch)).data.numpy()
            
            # go through all mini batch entries
            for j in range(0, outp.shape[0]):
                
                # find maximally active unit for entry
                if outp.ndim == 2: # flattened layer
                    ua.append([0, np.argmax(outp[j,:]), i*outp.shape[0]+j, outp[j,:].max()])
                else: # unflattened
                    idx = np.argmax(outp[j,fm,:,:], axis=None)
                    multi_idx = np.unravel_index(idx, outp[j,fm,:,:].shape)
                    ua.append([*multi_idx, i*outp.shape[0]+j, outp[j,fm,:,:].max()])
                
                # sort and resize
                ua.sort(key=lambda x: float(x[3]), reverse=True)
                ua = ua[:n]
        
        return ua
    
    
    # find input block for all maxima
    def _find_input_block(self, fm, ua):
        '''
        Internal function: finds the input block for all maxima.
        
        Arguments:
        
        * fm: feature map index of interest in the current output layer.
        * ua: list of lists with position and image index of all maxima (as returned by 
          _get_max_units()).
        
        Returns a list of block sizes (6-tuples) and the maximum block size in z,y, and
        x-direction.
        '''
        blocks = []
        dz = dy = dx = 1
        for i in range(0, len(ua)):
            
            if ua[i][3] == -math.inf:
                continue
            
            # init with position of unit
            bl = [fm, ua[i][0], ua[i][1], fm, ua[i][0], ua[i][1]] 
            
            # go back through path to unflattened input and get block size
            for key, module in reversed(self._testnet._modules.items()):
                
                # stop if the layer is Unflatten
                if module.__class__.__name__ == 'Unflatten': 
                    break
                
                # skip if layer is an activation function
                if module.__class__.__name__ in self._act_funcs:
                    continue
                
                bl = module.get_input_block(bl)
            
            # update maximal block size
            if bl[3] - bl[0] + 1 > dz:
                dz = bl[3] - bl[0] + 1
            if bl[4] - bl[1] + 1 > dy:
                dy = bl[4] - bl[1] + 1
            if bl[5] - bl[2] + 1 > dx:
                dx = bl[5] - bl[2] + 1
            
            # append to block list
            blocks.append(bl)
        
        return blocks, dz, dy, dx
    
    
    # replace ReLU for guided backprop
    def _replace_relus(self):
        
        # set negative part to 0
        def relu_hook(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # If layer is ReLU, do hookup clamping as postprocessing step
        for key, module in self._testnet._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook)
    
    
    # Get the n input regions that activate a feature map most
    def get_max_inputs(self, layer, fm, n, dataset, batch_size = -1):
        '''
        Finds the n input regions that activate a feature map 'fm' most::
        
            max_inps = vis.get_max_inputs('Name of layer', fm, n, X_test)
        
        This method finds the 'n' input regions that maximally activate a specified
        feature map 'fm' in a given network layer for a test set of inputs 'X_test'. The 
        method works only for networks with an 'Unflatten' layer as input and a single branch, e.g. a
        convolutional network. The test input must be a flattened and fit the input size of the network. 
        The result comes in the form of a 4-tensor where the first dimension 
        is n, the second the number of input channels that feed into the feature map, and 
        the last dimensions are the size of the input regions. The tensor contains the 
        desired n image patches.
        '''
        # check parameters
        self._build_subnet(layer)
        dataloader = self._prepare_dataset(dataset, batch_size)
        fm = int(fm)
        if self._outsize[0] <= fm:
            raise ValueError('Feature map index exceeds limits.')
        n = int(n)
        if n < 1:
            raise ValueError('Number of input regions must be > 0.')
        
        # Find the n inputs that maximally activate the units in the feature map
        ua = self._get_max_units(fm, n, dataloader)
        
        # find input block for all maxima
        blocks,dz,dy,dx = self._find_input_block(fm, ua)
        
        # output array
        outp = np.zeros((len(blocks), dz, dy, dx), dtype=np.float32)
        for i in range(0, len(blocks)):
            
            # get image and unflatten it
            img_flat = (dataloader.dataset[ua[i][2]]).data.numpy()
            img = np.reshape(img_flat, self._insize)
            
            # copy image block into output tensor
            bl = blocks[i]
            outp[i,:bl[3]-bl[0]+1,:bl[4]-bl[1]+1,:bl[5]-bl[2]+1] = \
                img[bl[0]:bl[3]+1,bl[1]:bl[4]+1,bl[2]:bl[5]+1]
        
        return outp
    
    
    # saliency map
    def saliency(self, layer, fm, n, dataset, batch_size = -1, guided_backprop=True):
        '''
        Get the error signal for activating the n most active feature map units::
        
            sal = vis.saliency(name_of_layer, fm, n, dataset, batch_size, False)
        
        This method finds the error signal for the 'n' input regions that maximally 
        activate a specified feature map 'fm' in a given network layer for a test set of 
        inputs 'dataset'. The method is described in Simonyan et al. (2013). If the flag
        'guided_backprop' is set to True (default) then only the positive error signal is
        backpropagated through the ReLU layers in the network. This method is called
        "Guided Backpropagation" (Springenberg et al., 2015) and leads to much clearer
        saliency maps as the plain error signal proposed by Simonyan et al. (2013) which
        you get by setting 'guided_backprop' to False. If desired, you can specify a batch 
        size for evaluating the test set.
        
        The method works only for networks with an 'Unflatten' layer as input, e.g. a
        convolutional network. The test input must be a flattened and fit the input size of the network. 
        The result comes in the form of a 4-tensor where the first dimension 
        is n, the second the number of input channels that feed into the feature map, and 
        the last dimensions are the size of the unflattened input images. The tensor contains the 
        desired n error signals which form a kind of saliency map for the input image.
        '''
        
        # backward hook to be registered at first layer: saves the gradient
        def hook_function(module, grad_in, grad_out):
            self._gradient = grad_out[0]
        
        # check parameters
        self._build_subnet(layer)
        dataloader = self._prepare_dataset(dataset, batch_size)
        fm = int(fm)
        if self._outsize[0] <= fm:
            raise ValueError('Feature map index exceeds limits.')
        n = int(n)
        if n < 1:
            raise ValueError('Number of input regions must be > 0.')
        
        # register backward hook at first layer
        (self._testnet)[0].register_backward_hook(hook_function)
        
        # register modified ReLus for guided backprop
        if guided_backprop:
            self._replace_relus()
        
        # Find the n inputs that maximally activate the units in the feature map
        ua = self._get_max_units(fm, n, dataloader)
        
        # alloc output tensor (n x same size as unflattened input)
        outp = np.zeros((len(ua), self._insize[0], self._insize[1], self._insize[2]), 
            dtype=np.float32)
        
        # fill it with saliency maps
        for i in range(0, len(ua)):
            
            # skip if no valid max was found
            if ua[i][3] == -math.inf:
                continue
            
            # get position of maximum and associated image index
            ypos = ua[i][0]
            xpos = ua[i][1]
            img_ndx = ua[i][2]
            
            # get flattened input image
            img_flat = Variable(dataloader.dataset[img_ndx], requires_grad=True)
            
            # generate an empty image the size of the network output and set max to 1
            if self._outsize[0] == 1 and self._outsize[1] == 1: # flattened output layer
                one_hot_output = torch.FloatTensor(1, self._outsize[2]).zero_()
                one_hot_output[0][xpos] = 1.0
            else: # unflattened output layer
                one_hot_output = torch.FloatTensor(1, self._outsize[0], self._outsize[1], 
                    self._outsize[2]).zero_()
                one_hot_output[0][fm][ypos][xpos] = 1.0
            
            # compute model output for provided image
            model_output = self._testnet(img_flat)
            
            # backward pass
            self._testnet.zero_grad()
            model_output.backward(gradient=one_hot_output)
            np_gradient = self._gradient.data.numpy()[0]
            
            # copy gradient into output image
            outp[i,:,:,:] = np_gradient
            
        return outp
    
    
    # get positive and negative saliency
    def signed_saliency(self, gradient):
        '''
        Decomposes a saliency map into a positive and a negative saliency::
        
            pos_saliency, neg_saliency = vis.signed_saliency(saliency)
        
        '''
        pos_saliency = (np.maximum(0, gradient) / gradient.max())
        neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
        
        return pos_saliency, neg_saliency


if __name__ == "__main__":
    print("This is", Visualizer().name)
