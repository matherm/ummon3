import math
import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Crop', 'Whiten', 'RandomFlipLR', 'RandomBrightness', 'RandomContrast' ]

# image cropping
class Crop(nn.Module):
    '''
    Random image cropping::
    
        cro    = Crop([n,p,q], ycrop, xcrop)
    
    This layer cuts out a specified window of size 'ycrop' x 'xcrop' image-wise from the 
    input tensor of size [:,n,p,q]. The position of the window is random during training 
    and central during prediction and loss computation. The number of channels remains 
    unchanged. 
    
    Attributes:
    
    * ycrop: cropping window height (read only)
    * xcrop: cropping window width (read only)
    
    '''
    # constructor
    def __init__(self, insize, xcrop=3, ycrop=3):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        ycrop = int(ycrop)
        xcrop = int(xcrop)
        if ycrop < 1 or xcrop < 1:
            raise ValueError('Cropping window must be greater than zero.')
        if ycrop >= self.insize[1] or xcrop >= self.insize[2]:
            raise ValueError('Cropping window must be smaller than input size.')
        self._ycrop = ycrop
        self._xcrop = xcrop
        
        super(Crop, self).__init__()
        
        # add layer to network
        self.outsize = [self.insize[0], self.ycrop, self.xcrop]
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + '[bs,{},{},{}]->[bs,{},{},{}];ycr={};xcr={})'.format(
            self.insize[0], self.insize[1], self.insize[2], self.outsize[0], self.outsize[1], 
            self.outsize[2],self.ycrop, self.xcrop)
    
    
    # Forward path
    def forward(self, input):
        
        # not in training mode: cut out central window
        if not self.training:
            y0 = (self.insize[1] - self.ycrop)//2
            x0 = (self.insize[2] - self.xcrop)//2
            return input[:,:, y0:y0+self.ycrop, x0:x0+self.xcrop]
            
        else: # training mode: random window
            output = torch.zeros(input.size()[0], *self.outsize, dtype=input.dtype)
            for i in range(input.size()[0]): # go through mini batch and select varying subwindow
                y0 = int(torch.randint(low=0, high=self.insize[1]-self.ycrop+1, size=(1,)).item())
                y1 = y0 + self.ycrop
                x0 = int(torch.randint(low=0, high=self.insize[2]-self.xcrop+1, size=(1,)).item())
                x1 = x0 + self.xcrop
                output[i, :, :, :] = input[i,:, y0:y1, x0:x1]
            return output
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        y0 = (self.insize[1] - self.ycrop)//2
        x0 = (self.insize[2] - self.xcrop)//2
        return [outp[0], outp[1] + y0, outp[2] + x0, outp[3], outp[4] + y0, outp[5] + x0]
    
    
    # Attribute: cropping window height (read only)
    @property
    def ycrop(self):
        return self._ycrop
    
    # Attribute: cropping window width (read only)
    @property
    def xcrop(self):
        return self._xcrop


# image whitening
class Whiten(nn.Module):
    '''
    Approximate image whitening::
    
        white  = Whiten([n,p,q])
    
    Linearly scales input tensor of size [:,n,p,q] image-wise to have zero mean and unit 
    norm. This layer computes (x - mean) / dev, where dev = max(stddev, 
    1.0/sqrt(number of pixels). The standard deviation is capped away from zero to protect 
    against division by 0 when handling uniform images. Input and output size are the same.
    '''
    # constructor
    def __init__(self, insize):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        super(Whiten, self).__init__()
        self.outsize = self.insize
        
        # minimum variance for normalization
        self._mindev = np.float32(1.0/math.sqrt(self.insize[1]*self.insize[2]))
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + '[bs,{},{},{}]->[bs,{},{},{}])'.format(self.insize[0], self.insize[1], 
            self.insize[2], self.outsize[0], self.outsize[1], self.outsize[2])
    
    # Forward path
    def forward(self, input):
        output = torch.zeros(input.size()[0], *self.outsize, dtype=input.dtype)
        for i in range(input.size()[0]):
            for j in range(self.insize[0]):
                img = input[i,j,:,:]
                m = img.mean().item()
                s = torch.std(img).item()
                if s < self._mindev:
                    s = self._mindev
                s = 1.0/s
                centred_img = torch.add(img, -m)
                output[i,j,:,:] = torch.mul(centred_img, s)
        return output
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp


# random left right flipping of an image
class RandomFlipLR(nn.Module):
    '''
    Randomly flips an image horizontally (left to right):: 
    
        flip  = RandomFlipLR([n,p,q])
    
    With a 1 in 2 chance, outputs the contents of an input tensor of size [:,n,p,q] with
    its images horizontally flipped; otherwise outputs the images as-is. Input and output 
    size are the same. Active only during training, otherwise only copies input to output.
    '''
    # constructor
    def __init__(self, insize):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        super(RandomFlipLR, self).__init__()
        self.outsize = self.insize
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + '[bs,{},{},{}]->[bs,{},{},{}])'.format(self.insize[0], self.insize[1], 
            self.insize[2], self.outsize[0], self.outsize[1], self.outsize[2])
    
    
    # flip a tensor (Cuda + CPU)
    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
            -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)
    
    
    # Forward path
    def forward(self, input):
        
         # not in training mode: do nothing
        if not self.training:
            return input
        else: # training: random flip
            output = torch.zeros(input.size()[0], *self.outsize, dtype=input.dtype)
            for i in range(input.size()[0]):
                rand = int(torch.randint(low=0, high=2, size=(1,)).item())
                for j in range(self.insize[0]):
                    if rand > 0:
                        img = self.flip(input[i,j,:,:], 1)
                    else:
                        img = input[i,j,:,:]
                    output[i,j,:,:] = img
            return output
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp


# random brightness adjustment of an image
class RandomBrightness(nn.Module):
    '''
    Adjust image brightness randomly::
    
        bright = RandomBrightness([n,p,q], max_delta)
    
    Adds a random number to all channels of the images in the input tensor during 
    training, thereby adjusting the overall brightness of the image. The random number is 
    drawn uniformly from the interval [-max_delta, max_delta] specified in the constructor. 
    No clipping is done, so this node should be used together with Whiten. Input and 
    output size are the same. Active only during training, otherwise only copies input to 
    output. 
    
    Attributes:
    
    * max_delta: half width of additive brightness range (read only)
    
    '''
    # constructor
    def __init__(self, insize, max_delta=0.5):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        max_delta = float(max_delta)
        if max_delta <= 0:
            raise ValueError('Maximum brightness adjustment must be greater than zero.')
        self._max_delta = max_delta
        super(RandomBrightness, self).__init__()
        self.outsize = self.insize
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + '[bs,{},{},{}]->[bs,{},{},{}],delta={:.1f})'.format(self.insize[0], self.insize[1], 
            self.insize[2], self.outsize[0], self.outsize[1], self.outsize[2], self.max_delta)
    
    
    # Forward path
    def forward(self, input):
        '''
        Add random constant to image.
        '''
        if not self.training:
            return input
        else:
            output = torch.zeros(input.size()[0], *self.outsize, dtype=input.dtype)
            for i in range(input.size()[0]):            
                rand = 2.0*self.max_delta*(torch.rand(size=(1,)).item() - 0.5)
                for j in range(self.insize[0]):
                    output[i,j,:,:] = torch.add(input[i,j,:,:], rand)
            return output
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp
    
    
    # Attribute: half width of additive brightness range (read only)
    @property
    def max_delta(self):
        return self._max_delta


# random contrast adjustment of an image
class RandomContrast(nn.Module):
    '''
    Adjust image contrast randomly::
    
        bright = RandomContrast([n,p,q], lower, upper)
    
    Randomly adjusts the contrast of each image channel during training according to
    (x - mean)*contrast_factor + mean. The contrast factor is drawn uniformly from the 
    interval [lower, upper] specified in the constructor. No clipping is done, so this 
    node should be used together with Whiten. Input and output size are the same. Active 
    only during training, otherwise only copies input to output. 
    
    Attributes:
    
    * lower: lowest contrast factor (read only)
    * upper: highest contrast factor (read only)
    
    '''
    # constructor
    def __init__(self, insize, lower=0.9, upper=1.1):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        lower = float(lower)
        upper = float(upper)
        if lower < 0:
            raise ValueError('Lower bound must be >= 0.')
        if upper < 0:
            raise ValueError('Upper bound must be >= 0.')
        if upper <= lower:
            raise ValueError('Upper bound must be > lower bound.')
        self._lower = lower
        self._upper = upper
        super(RandomContrast, self).__init__()
        self.outsize = self.insize
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + '[bs,{},{},{}]->[bs,{},{},{}],c=[{:.3f},{:.3f}])'.format(self.insize[0], self.insize[1], 
            self.insize[2], self.outsize[0], self.outsize[1], self.outsize[2], self.lower, self.upper)
    
    
    #  Forward path
    def forward(self, input):
        if not self.training:
            return input
        else:
            output = torch.zeros(input.size()[0], *self.outsize, dtype=input.dtype)
            for i in range(input.size()[0]):            
                contrast_fac = self.lower + (self.upper - self.lower)*torch.rand(size=(1,)).item()
                for j in range(self.insize[0]):
                    m = input[i,j,:,:].mean()
                    img = torch.add(torch.mul(torch.add(input[i,j,:,:], -m), contrast_fac), m)
                    output[i,j,:,:] = img
            return output
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp
    
    
    # Attribute: lowest contrast factor (read only)
    @property
    def lower(self):
        return self._lower
    
    
    # Attribute: highest contrast factor (read only)
    @property
    def upper(self):
        return self._upper

